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

from xmixers.utils import XMIXERS_DEBUG, print_params


class SRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x) * self.weight

        return output

    def extra_repr(self):
        return print_module(self)


class GatedRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8, bias: bool = False) -> None:
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

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        return self.scale * x_normed * torch.sigmoid(self.gate * x)
