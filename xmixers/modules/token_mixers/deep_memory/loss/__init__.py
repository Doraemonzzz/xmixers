import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_fn(loss_name):
    if loss_name == "mse":

        def fn(y_pred, y_true):
            return F.mse_loss(y_pred, y_true, reduction="sum")

    elif loss_name == "inner_product":

        def fn(y_pred, y_true):
            return -torch.sum((y_pred * y_true).sum(dim=-1))

    else:
        raise ValueError(f"Loss function {loss_name} not found")

    return fn
