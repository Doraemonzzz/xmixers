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
from xopes.ops.normalize import rmsnorm_fn

from xmixers.utils import XMIXERS_DEBUG, print_module, print_params


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def extra_repr(self) -> str:
        return print_module(self)

    def forward(self, x, residual=None, return_residual=False):
        return rmsnorm_fn(
            x=x,
            weight=self.weight,
            dim=self.dim,
            eps=self.eps,
            residual=residual,
            return_residual=return_residual,
        )


class GatedRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6, bias: bool = False, **kwargs) -> None:
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
        self.register_parameter("scale", self.scale)

    def extra_repr(self) -> str:
        return print_module(self)

    def forward(self, x):
        # TODO: add fusion here
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        return self.scale * x_normed * torch.sigmoid(self.gate * x)
