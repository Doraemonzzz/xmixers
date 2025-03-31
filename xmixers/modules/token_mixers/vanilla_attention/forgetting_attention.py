# flex attention: https://pytorch.org/blog/flexattention/
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache
from xopes.ops.flash_attn import forgetting_attn_fn

from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params


class ForgettingAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_heads: int = -1,
        bias: bool = False,
        layer_idx: int = 0,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        window_size: int = -1,
        init_std: float = 0.02,
        gain: float = 0.01,
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
        if self.kv_heads == -1:
            kv_dim = embed_dim
        else:
            kv_dim = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.f_proj = nn.Linear(embed_dim, num_heads, bias=bias)

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

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        b, n, d = x.shape
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        log_f = F.logsigmoid(self.f_proj(x))

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v],
        )

        if past_key_values is not None:
            past_key_values.get_seq_length(self.layer_idx)

        # cache update
        if past_key_values is not None:
            k, v, log_f = past_key_values.update(
                attn_state=(k, v, log_f),
                layer_idx=self.layer_idx,
                offset=n,
            )["attn_state"]

        causal = True if self.training or q.shape[-3] == k.shape[-3] else False

        scale = self.head_dim**-0.5
        head_first = True
        q, k, v = map(lambda x: rearrange(x, "... n h d -> ... h n d"), [q, k, v])
        log_f = rearrange(log_f, "... n h -> ... h n")

        if attention_mask is not None and not attention_mask.all():
            seq_start = n - attention_mask.sum(dim=-1)
            output = forgetting_attn_fn(
                q=q,
                k=k,
                v=v,
                log_fgate=log_f,
                head_first=head_first,
                seq_start=seq_start,
                sm_scale=scale,
            )
        else:
            output = forgetting_attn_fn(
                q=q,
                k=k,
                v=v,
                log_fgate=log_f,
                head_first=head_first,
                sm_scale=scale,
            )
        output = rearrange(output, "... h n d -> ... n h d")

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        # outproj
        output = self.o_proj(output)

        return output, past_key_values
