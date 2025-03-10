"""
Tpe in https://arxiv.org/abs/2405.21022
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache
from xopes.ops import lightning_attn_func

from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import _initialize_weights, print_module


class Tpe(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
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
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        self.q = nn.Parameter(torch.randn(embed_dim))
        self.k = nn.Parameter(torch.randn(embed_dim))
        self.log_decay = nn.Parameter(torch.randn(num_heads))
        self.norm = get_norm_fn(token_mixer_norm_type)(
            embed_dim, bias=bias, num_groups=num_heads
        )
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.layer_idx = layer_idx

        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self._init_weights()

    def _init_weights(self):
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        return _initialize_weights(self, module)

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        b, n, d = x.shape
        q, k, v = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim),
            [self.q, self.k, x],
        )
        log_decay = F.logsigmoid(self.log_decay)

        recurrent_state = None
        q_offset = 0
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"][0]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )

        # TODO: update this later
        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1).unsqueeze(-1)
            v = v.masked_fill(attention_mask_ == 0, 0)

        output, recurrent_state = lightning_attn_func(
            q=q,
            k=k,
            v=v,
            ld=log_decay,
            initial_state=recurrent_state,
        )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=[recurrent_state],
                layer_idx=self.layer_idx,
                offset=n,
            )

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")

        # normalize
        output = self.norm(output)

        # outproj
        output = self.o_proj(output)

        return output, past_key_values
