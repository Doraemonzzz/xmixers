# stick-breaking attention: https://arxiv.org/abs/2410.17980 and https://github.com/IBM/dolomite-engine/blob/main/dolomite_engine/hf_models/modeling_utils/sequence_mixer_blocks/stickbreaking_attention.py
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from transformers.cache_utils import Cache

from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params

try:
    from stickbreaking_attention.sb_attn import sb_attn
    from stickbreaking_attention.sb_varlen import sb_attn_varlen
except:
    sb_attn = None
    sb_attn_varlen = None


def decoding_stickbreaking(q, k, v, scale=None):
    """
    Stick-breaking attention weights.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    assert q.shape[2] == 1
    original_dtype = q.dtype
    q = q.float()
    k = k.float()
    logits = q @ k[..., :-1, :].transpose(-1, -2) * scale
    log_z = F.logsigmoid(logits).to(original_dtype)
    log_beta = F.logsigmoid(-logits).to(original_dtype)
    re_cum_log_beta = log_beta.flip(-1).cumsum(dim=-1).flip(-1) - log_beta
    log_att = log_z + re_cum_log_beta
    att: torch.Tensor = log_att.exp()
    v = v[..., :-1, :]
    out = torch.einsum("bhij,bhjd->bhid", att, v)

    return out, 1 - att.sum(dim=-1)


def sb_attn(q, k, v, mask=None, cum_weight=None):
    """
    Stick-breaking attention weights.
    """
    n = q.shape[1]
    if mask is None:
        mask = torch.ones(n, n).triu(0).to(q).bool()
    if cum_weight is None:
        cum_weight = torch.ones(n, n).tril(-1).to(q)

    scale = q.shape[-1] ** -0.5

    logits = (q @ k.transpose(-1, -2)) * scale

    original_dtype = logits.dtype

    log_z = F.logsigmoid(logits).masked_fill(mask, -1e5).to(original_dtype)
    log_beta = F.logsigmoid(-logits).masked_fill(mask, 0).to(original_dtype)

    re_cum_log_beta = torch.einsum("bhij,jk->bhik", log_beta, cum_weight.to(log_beta))
    log_att = log_z + re_cum_log_beta
    att = log_att.exp()
    o, rem = att @ v, 1 - att.sum(dim=-1)
    o = o + rem[..., None] * v

    return o, None


class StickBreakingAttention(nn.Module):
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

        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self.mask = torch.empty(0)
        self.cum_weight = torch.empty(0)
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

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v],
        )

        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        # cache update
        if past_key_values is not None:
            k, v = past_key_values.update(
                attn_state=(k, v),
                layer_idx=self.layer_idx,
                offset=n,
            )["attn_state"]

        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )
        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1).unsqueeze(-1)
            v = v.masked_fill(attention_mask_ == 0, 0)

        if self.mask.shape[0] < n:
            self.mask = torch.ones(n, n).triu(0).to(q).bool()
            self.cum_weight = torch.ones(n, n).tril(-1).to(q)

        q, k, v = map(lambda x: rearrange(x, "... n h d -> ... h n d"), [q, k, v])
        if q.shape[2] == k.shape[2]:
            output, _ = sb_attn(
                q=q,
                k=k,
                v=v,
                mask=self.mask,
                cum_weight=self.cum_weight,
            )
        else:
            output, _ = decoding_stickbreaking(
                q=q,
                k=k,
                v=v,
            )
        output = rearrange(output, "... h n d -> ... n h d")

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")

        # outproj
        output = self.o_proj(output)

        return output, past_key_values
