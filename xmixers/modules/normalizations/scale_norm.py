"""
ScaleNorm in https://arxiv.org/pdf/2202.10447.pdf
"""

import torch
from torch import nn

from xmixers.utils import XMIXERS_DEBUG, print_params


class ScaleNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.d = d
        self.eps = eps
        self.scala = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean_square = (x**2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps) * self.scala
        return x
