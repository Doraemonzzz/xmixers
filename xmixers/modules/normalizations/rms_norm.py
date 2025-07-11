"""
SimpleRMSNorm in https://arxiv.org/abs/2307.14995
RMSNorm in https://arxiv.org/pdf/1910.07467.pdf
GatedRMSNorm in https://arxiv.org/pdf/2104.07012.pdf

Reference:
https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
https://github.com/bzhangGo/zero/blob/master/modules/rela.py
"""
import torch
import torch.nn as nn
from fla.modules import RMSNorm

from xmixers.utils import XMIXERS_DEBUG, print_module, print_params

from .utils import NormOp


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.op = NormOp(norm_type="rmsnorm")

        self._init_weights()

    def _init_weights(self):
        nn.init.ones_(self.weight)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def forward(self, x, residual=None, return_residual=False):
        return self.op(
            x,
            self.weight,
            None,
            residual,
            self.dim,
            self.eps,
            False,
            1,
            return_residual,
        )

        return o


class RMSNormFusedGate(torch.nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, gate_act="sigmoid", gate_pos="pre", **kwargs
    ):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.gate_act = gate_act
        self.gate_pos = gate_pos
        self.op = NormOp(norm_type="rmsnorm")

        self._init_weights()

    def _init_weights(self):
        nn.init.ones_(self.weight)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, gate_act={self.gate_act}, gate_pos={self.gate_pos}"

    def forward(self, x, gate):
        return self.op(
            x,
            self.weight,
            None,
            None,
            self.dim,
            self.eps,
            False,
            1,
            False,
            gate,
            self.gate_act,
            self.gate_pos,
        )

        return o


class GatedRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, bias: bool = False, **kwargs) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.eps = eps
        self.d = d
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        self.gate = nn.Parameter(torch.ones(d))
        self.register_parameter("gate", self.scale)

        self._init_weights()

    def _init_weights(self):
        nn.init.ones_(self.scale)
        nn.init.ones_(self.gate)

    def extra_repr(self) -> str:
        return print_module(self)

    def forward(self, x):
        # TODO: add fusion here
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        return self.scale * x_normed * torch.sigmoid(self.gate * x)
