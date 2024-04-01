import torch.nn.functional as F
from torch import nn

from .utils import logger


def get_activation_fn(activation: str) -> None:
    logger.info(f"activation: {activation}")
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":
        return torch.exp
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":

        def f(x):
            return 1 + F.elu(x)

        return f
    elif activation == "2+elu":

        def f(x):
            return 2 + F.elu(x)

        return f
    elif activation == "silu":
        return F.silu
    else:
        return lambda x: x


class ActLayer(nn.Module):
    def __init__(
        self,
        activation: str,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.f = get_activation_fn(activation)

    def forward(self, x):
        return self.f(x)

    def extra_repr(self):
        return self.activation.lower()
