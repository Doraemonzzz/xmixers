# Tnl: https://arxiv.org/pdf/2405.17381
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache
from xopes.ops import lasd_fn

from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import (
    XMIXERS_DEBUG,
    _initialize_weights,
    _upad_input,
    pad_input,
    print_params,
)


class TnlAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        norm_type: str = "rmsnorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        v_activation: str = "silu",
        q_norm: bool = False,
        k_norm: bool = False,
        v_norm: bool = False,
        causal: bool = True,
        max_position_embeddings: int = 1024,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
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
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.norm = get_norm_fn(norm_type)(embed_dim)
        self.q_act = q_activation
        self.k_act = k_activation
        self.v_act = v_activation
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.v_norm = v_norm
        self.causal = causal

        self.use_output_gate = use_output_gate
        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )

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
        log_decay,
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

        recurrent_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"][0]
            past_key_values.get_seq_length(self.layer_idx)

        use_attn_mask = attention_mask is not None and not attention_mask.all()

        if use_attn_mask:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
                q=q, k=k, v=v, attention_mask=attention_mask, q_len=n
            )
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            cu_seqlens, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
        else:
            cu_seqlens = None

        if self.causal:
            output, recurrent_state = lasd_fn(
                q=q,
                k=k,
                v=v,
                ld=log_decay,
                initial_state=recurrent_state,
                cu_seqlens=cu_seqlens,
                q_act=self.q_act,
                k_act=self.k_act,
                v_act=self.v_act,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                v_norm=self.v_norm,
            )
        else:
            assert False, "not implemented"

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=[recurrent_state],
                layer_idx=self.layer_idx,
                offset=n,
            )

        if use_attn_mask:
            output = pad_input(output.squeeze(0), indices_q, b, n)

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")

        if self.use_output_gate:
            output_gate = F.sigmoid(self.output_gate(x))
            output = output * output_gate

        output = self.norm(output)

        # outproj
        output = self.out_proj(output)

        return output, past_key_values
