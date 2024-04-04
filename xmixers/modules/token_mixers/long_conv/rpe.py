"""
Relative Position Encoder in https://arxiv.org/pdf/2305.04749.pdf
"""

import torch
import torch.nn as nn

from xmixers.modules import get_norm_fn
from xmixers.modules.activations import ActLayer
from xmixers.utils import XMIXERS_DEBUG, print_params


class Rpe(nn.Module):
    def __init__(
        self,
        in_dim: int,
        feature_dim: int,
        out_dim: int,
        activation: str = "silu",
        bias: bool = False,
        rpe_layers: int = 3,
        norm_type: str = "layernorm",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.in_dim = in_dim
        if in_dim > 1:
            theta = 10000 ** (-2 / in_dim * torch.arange(in_dim // 2)).reshape(1, -1)
            self.register_buffer("theta", theta)

        self.pos_proj = nn.Linear(in_dim, feature_dim, bias=bias)
        self.layers = nn.ModuleList([])
        for _ in range(rpe_layers):
            self.layers.append(
                nn.Sequential(
                    get_norm_fn(norm_type)(feature_dim),
                    ActLayer(activation),
                    nn.Linear(feature_dim, feature_dim, bias=bias),
                )
            )
        self.out = nn.Sequential(
            get_norm_fn(norm_type)(feature_dim),
            ActLayer(activation),
            nn.Linear(feature_dim, out_dim, bias=bias),
        )

    def get_feature(self, index):
        if self.in_dim > 1:
            theta = index * self.theta
            x = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        else:
            x = index

        return x

    def forward(self, index):
        input = self.get_feature(index).to(self.pos_proj.weight.dtype)
        x = self.pos_proj(input)
        for m in self.layers:
            x = m(x) + x
        x = self.out(x)

        return x
