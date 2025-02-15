import torch.nn as nn
from xopes.ops import gate_linear_fn


class GateLinearOp(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x1, x2, weight, bias, act, residual=None):
        output = gate_linear_fn(
            x1=x1,
            x2=x2,
            W=weight,
            bias=bias,
            act=act,
            residual=residual,
        )
        return output
