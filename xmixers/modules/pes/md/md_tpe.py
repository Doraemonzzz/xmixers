from typing import List, Optional

import torch
import torch.nn as nn
from einops import pack, unpack
from transformers.cache_utils import Cache

from ..tpe import Tpe


class MdTpe(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dims: List[int] = [-2],
        bias: bool = False,
        layer_idx: int = 0,
        token_mixer_norm_type: str = "rmsnorm",
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.dims = dims
        self.tpes = nn.ModuleList([])
        for dim in dims:
            self.tpes.append(
                Tpe(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dim=dim,
                    bias=bias,
                    layer_idx=layer_idx,
                    token_mixer_norm_type=token_mixer_norm_type,
                    token_mixer_init_type=token_mixer_init_type,
                    rescale_type=rescale_type,
                    num_layers=num_layers,
                    init_std=init_std,
                    gain=gain,
                    **kwargs,
                )
            )

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        o = 0
        for i, tpe in enumerate(self.tpes):
            o += self.forward_tpe(x, tpe, dim=self.dims[i])
        return o

    def forward_tpe(self, x, tpe, dim=-2):
        if dim != -2:
            x = x.transpose(dim, -2)

        x, ps = pack([x], "* n d")
        x = tpe(x)[0]
        x = unpack(x, ps, "* n d")[0]

        if dim != -2:
            x = x.transpose(dim, -2)

        return x
