"""
FFN in https://arxiv.org/pdf/2002.05202.pdf
"""

import torch.nn as nn

from xmixers.modules import BaseModule
from xmixers.modules.activations import get_activation_fn


class FFN(BaseModule):
    def __init__(
        self, embed_dim: int, mid_dim: int, activation: str, bias: bool = True
    ) -> None:
        super().__init__()

        self.w1 = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.w2 = nn.Linear(mid_dim, embed_dim, bias=bias)
        self.act = get_activation_fn(activation)

    def forward(self, x):
        output = self.w2(self.act(self.w1(x)))

        return output
