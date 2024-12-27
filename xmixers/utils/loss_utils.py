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
    from xopes.ops.cross_entropy import linear_cross_entropy_baseline
except:
    linear_cross_entropy_baseline = None

AUTO_LOSS_MAPPING = {
    "naive": nn.CrossEntropyLoss(),
    "fla_fce": FusedCrossEntropyLoss(inplace_backward=True),
    "cut_ce": linear_cross_entropy,
    "fla_flce": FusedLinearCrossEntropyLoss(),
    "liger_flce": LigerFusedLinearCrossEntropyLoss(),
    "lce_baseline": linear_cross_entropy_baseline,
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
    elif ce_type == "lce_baseline":
        return loss_fct(x=hidden_state, y=labels, At=weight.transpose(0, 1))
