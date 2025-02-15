# GLU: https://arxiv.org/pdf/2002.05202.pdf

import torch.nn as nn

from xmixers.modules.activations import get_activation_fn
from xmixers.utils import XMIXERS_DEBUG, print_params

from .utils import GateLinearOp


class GLU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mid_dim: int,
        activation: str,
        bias: bool = False,
        use_gate_linear: bool = False,
    ) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.w1 = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.w2 = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.w3 = nn.Linear(mid_dim, embed_dim, bias=bias)
        self.act = get_activation_fn(activation)
        self.activation = activation
        self.use_gate_linear = use_gate_linear
        self.gate_linear_op = GateLinearOp()

    def forward(self, x, residual=None):
        if self.use_gate_linear:
            # since we may use forward_pre_hook, we can't use key word arguments
            output = self.gate_linear_op(
                self.w1(x),
                self.w2(x),
                self.w3.weight,
                self.w3.bias,
                self.activation,
                residual,
            )
        else:
            output = self.w3(self.act(self.w1(x)) * self.w2(x))

        return output
