from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from xmixers.modules.pes import Lrpe
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None


import math

from xopes.ops.poly_attn import poly_attn_fn


def next_power_of_2_python(n):
    return 2 ** math.ceil(math.log2(n))


class PolyAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_heads: int = -1,
        bias: bool = False,
        use_lrpe: bool = True,
        layer_idx: int = 0,
        lrpe_type: int = 1,
        base: int = 10000,
        max_position_embeddings: int = 1024,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        window_size: int = -1,
        window_head_dim: int = 128,
        init_std: float = 0.02,
        gain: float = 0.01,
        poly_order: int = 4,
        chunk_size: int = 256,
        poly_type: int = 1,
        **kwargs,
    ):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.layer_idx = layer_idx
        self.kv_heads = kv_heads
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.window_head_dim = window_head_dim
        self.poly_type = poly_type
        if self.kv_heads == -1:
            kv_dim = embed_dim
        else:
            kv_dim = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                lrpe_type=lrpe_type,
                base=base,
            )

        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self.poly_order = poly_order
        self.chunk_size = chunk_size
        self.mask = torch.empty(0)
        self._init_weights()

    def _init_weights(self):
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        return _initialize_weights(self, module)

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        b, n, d = x.shape
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v],
        )

        # lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k, offset=q_offset)

        # cache update
        if past_key_values is not None:
            k, v = past_key_values.update(
                attn_state=(k, v),
                layer_idx=self.layer_idx,
                offset=n,
            )["attn_state"]

        causal = True if self.training or q.shape[-3] == k.shape[-3] else False

        if self.mask.shape[0] < n:
            l = next_power_of_2_python(n)
            self.mask = torch.tril(torch.ones(l, l).to(q.device))

        output = poly_attn_fn(
            q=q,
            k=k,
            v=v,
            p=self.poly_order,
            poly_type=self.poly_type,
            causal=causal,
            mask=self.mask[:n, :n],
            attention_mask=attention_mask,
            chunk_size=self.chunk_size,
        )
        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")

        if self.window_size > 0:
            window_size = (self.window_size, 0)
            g = self.window_head_dim // self.head_dim
            q, k, v = map(
                lambda x: rearrange(x, "... (h g) d -> ... h (g d)", g=g),
                [q, k, v],
            )
            output_swa = flash_attn_func(
                q, k, v, causal=causal, window_size=window_size
            )
            # reshape
            output_swa = rearrange(output_swa, "... n h d -> ... n (h d)")
            output = output + output_swa

        # outproj
        output = self.o_proj(output)

        return output, past_key_values
