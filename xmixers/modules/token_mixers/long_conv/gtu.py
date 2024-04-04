"""
Gated Toeplitz Unit in https://arxiv.org/pdf/2305.04749.pdf
Add support for multi dimension.
For example, if the input dim is (h, w, d),
then the output is g * (tno1(x) + tno2(x)).
"""

from typing import List

import torch.nn as nn

from xmixers.modules.activations import get_activation_fn
from xmixers.utils import XMIXERS_DEBUG, print_params

from .tno import Tno


class Gtu(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expand_ratio: int = 1,
        bias: bool = False,
        activation: str = "silu",
        causal: bool = True,
        norm_type: str = "layernorm",
        use_decay: bool = True,
        rpe_in_dim: int = 1,
        rpe_feature_dim: int = 32,
        rpe_layers: int = 3,
        dims: List[int] = [-2],
        lower_bound: float = 0.99,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio

        d1 = int(self.expand_ratio * embed_dim)
        # linear projection
        self.uv_proj = nn.Linear(embed_dim, 2 * d1, bias=bias)
        self.o = nn.Linear(d1, embed_dim, bias=bias)
        self.act = get_activation_fn(activation)
        self.dims = dims
        self.tno_list = nn.ModuleList([])
        for dim in dims:
            # tno
            self.tno_list.append(
                Tno(
                    in_dim=rpe_in_dim,
                    feature_dim=rpe_feature_dim,
                    out_dim=d1,
                    activation=activation,
                    bias=bias,
                    rpe_layers=rpe_layers,
                    norm_type=norm_type,
                    use_decay=use_decay,
                    causal=causal,
                    dim=dim,
                    lower_bound=lower_bound,
                )
            )

    def forward(self, x):
        # x: b, n, d
        u, v = self.act(self.uv_proj(x)).chunk(2, dim=-1)
        output = 0
        for tno in self.tno_list:
            output += tno(v)
        output = u * output

        output = self.o(output)

        return output
