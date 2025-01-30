import torch
import torch.nn as nn

try:
    from fla.modules import FusedLinearCrossEntropyLoss
except:
    FusedLinearCrossEntropyLoss = None

try:
    from fla.modules import FusedCrossEntropyLoss
except:
    FusedCrossEntropyLoss = lambda x: None

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except:
    LigerFusedLinearCrossEntropyLoss = lambda x: None

try:
    from cut_cross_entropy import linear_cross_entropy
except:
    linear_cross_entropy = None

try:
    from xopes.ops.cross_entropy import cross_entropy_fn
except:
    cross_entropy_fn = None

try:
    from xopes.ops.linear_cross_entropy import linear_cross_entropy_fn
except:
    linear_cross_entropy_fn = None

AUTO_LOSS_MAPPING = {
    "naive": nn.CrossEntropyLoss(),
    "fla_fce": FusedCrossEntropyLoss(inplace_backward=True),
    "cut_ce": linear_cross_entropy,
    "fla_flce": FusedLinearCrossEntropyLoss(),
    "liger_flce": LigerFusedLinearCrossEntropyLoss(),
    "xopes_ce": cross_entropy_fn,
    "xopes_flce": linear_cross_entropy_fn,
}


def loss_fct(ce_type, labels, logits=None, hidden_state=None, weight=None, bias=None):
    if ce_type not in AUTO_LOSS_MAPPING:
        raise ValueError(f"Loss function {ce_type} not found")
    loss_fct = AUTO_LOSS_MAPPING[ce_type]
    if ce_type == "naive":
        return loss_fct(logits, labels)
    elif ce_type == "fla_fce":
        return loss_fct(
            input=logits,
            target=labels,
        )
    elif ce_type == "fla_flce":
        return loss_fct(
            x=hidden_state,
            target=labels,
            weight=weight,
            bias=bias,
        )
    elif ce_type == "liger_flce":
        return loss_fct(
            lin_weight=weight.to(torch.bfloat16),
            _input=hidden_state.to(torch.bfloat16),
            target=labels,
            bias=bias,
        )
    elif ce_type == "cut_ce":
        return loss_fct(
            e=hidden_state.to(torch.bfloat16),
            c=weight.to(torch.bfloat16),
            targets=labels,
        )
    elif ce_type == "xopes_ce":
        return loss_fct(z=logits, y=labels)
    elif ce_type == "xopes_flce":
        return loss_fct(x=hidden_state, y=labels, W=weight, bias=bias)


def compute_loss(
    lm_head,
    ce_type,
    hidden_states,
    labels,
    num_logits_to_keep=0,
    fuse_linear_and_cross_entropy=True,
):
    logits = (
        None
        if fuse_linear_and_cross_entropy
        else lm_head(
            hidden_states[:, -num_logits_to_keep:]
        )  # when generation or prefilling, num_logits_to_keep is 1
    )

    loss = None
    if labels is not None:
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(hidden_states.device)

        if fuse_linear_and_cross_entropy:
            # Shift so that tokens < n predict n
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_hidden_states = shift_hidden_states.view(
                -1, shift_hidden_states.shape[-1]
            )
            loss = loss_fct(
                ce_type=ce_type,
                labels=shift_labels,
                hidden_state=shift_hidden_states,
                weight=lm_head.weight,
                bias=lm_head.bias,
            )
        else:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits[-1])
            loss = loss_fct(
                ce_type=ce_type,
                labels=shift_labels,
                logits=shift_logits,
            )

    return logits, loss
