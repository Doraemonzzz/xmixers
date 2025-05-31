import torch
import torch.nn as nn

from .loss import get_loss_fn
from .optimizer import get_optimizer


def get_chunk(x, i, chunk_size):
    if x is None:
        return None
    else:
        n = x.shape[1]
        start = i * chunk_size
        end = min(start + chunk_size, n)
        return x[:, start:end]


def stack_chunk(o):
    return torch.cat(o, dim=1)


def get_default_value(value_dict, keys, index, chunk_size=128):
    res_dict = {}
    for key in keys:
        if key in value_dict:
            res_dict[key] = get_chunk(value_dict[key], index, chunk_size)

    return res_dict


def prepare_fast_weight(fast_weight):
    # make sure all fast_weight parameters are leaf tensors
    for name, param in fast_weight.items():
        if not param.is_leaf:
            # create new leaf tensor
            new_param = param.detach().requires_grad_(True)
            fast_weight[name] = new_param
        else:
            param.requires_grad_(True)

    return fast_weight


def fast_weight_train(
    x_val: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    fast_weight_model: nn.Module,
    hyper_params_dict: dict,
    # optimizer
    optimizer_name: str = "sgd",
    lr: float = 1,
    wd: float = 0,
    momentum: float = 0,
    damping: float = 0,
    pooling_method: str = "mean",
    # loss
    loss_name: str = "mse",
    chunk_size: int = 128,
):
    b, n = x_val.shape[0], x_val.shape[1]
    # prepare the fast weight
    fast_weight = fast_weight_model.init_fast_weight(b)
    fast_weight = prepare_fast_weight(fast_weight)
    # prepare the optimizer
    optimizer = get_optimizer(
        optimizer_name,
        fast_weight=fast_weight,
    )
    # prepare the loss function
    loss_fn = get_loss_fn(loss_name)

    # train
    y_val = []
    num_chunks = (n + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        x_val_chunk = get_chunk(x_val, i, chunk_size)
        x_chunk = get_chunk(x, i, chunk_size)
        y_chunk = get_chunk(y, i, chunk_size)

        hyper_params_chunk = {
            name: get_default_value(
                hyper_params_dict[name], fast_weight.keys(), i, chunk_size
            )
            for name in hyper_params_dict.keys()
        }

        y_val_chunk = fast_weight_model.forward(x_val_chunk, fast_weight)
        y_val.append(y_val_chunk)

        # update the fast weight
        optimizer.zero_grad()

        with torch.enable_grad():
            y_chunk_pred = fast_weight_model.forward(x_chunk, fast_weight)
            loss = loss_fn(y_chunk_pred, y_chunk)
            loss.backward(retain_graph=True)
            optimizer.step(**hyper_params_chunk)

    y_val = stack_chunk(y_val)

    return y_val, fast_weight
