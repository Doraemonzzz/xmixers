"""
Offset scale: y = gamma * x + beta
"""

import torch
import torch.nn as nn

from xmixers.utils import XMIXERS_DEBUG, print_params


class OffsetScale(nn.Module):
    def __init__(self, dim: int, **kwargs) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = torch.einsum("... d, d -> ... d", x, self.gamma) + self.beta
        return out
