# multi product attention
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.pes import Lrpe
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except:
    flash_attn_func = None
    index_first_axis = None
    pad_input = None
    unpad_input = None

from .utils import _upad_input


class MultiProductAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
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
        mpa_type: int = 0,
        mpa_activation: str = "sigmoid",
        head_dim: int = -1,
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
        self.head_dim = embed_dim // num_heads if head_dim == -1 else head_dim
        self.window_size = window_size
        mid_dim = self.head_dim * self.num_heads
        self.mpa_type = mpa_type
        self.q_proj = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, 2 * self.head_dim, bias=bias)
        if self.mpa_type == 0:
            self.kv_head_proj = nn.Linear(embed_dim, 2 * num_heads, bias=bias)
        else:
            self.kv_head = nn.Parameter(
                torch.randn(2 * num_heads) * 0.1, requires_grad=True
            )
        self.act = get_activation_fn(mpa_activation)
        self.o_proj = nn.Linear(mid_dim, embed_dim, bias=bias)

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
        self.apply(self._initialize_weights)

    def extra_repr(self):
        return print_module(self)

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
        # x: b n d
        # linear map
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        if self.mpa_type == 0:
            kv_head = self.act(self.kv_head_proj(x))
        else:
            kv_head = self.act(self.kv_head)

        k, v = kv.chunk(2, dim=-1)
        k_head, v_head = kv_head.chunk(2, dim=-1)

        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads)

        # for lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k, offset=q_offset)

        k, v = map(
            lambda arr: torch.einsum("... h, ... d -> ... h d", arr[0], arr[1]),
            [(k_head, k), (v_head, v)],
        )

        # cache update
        if past_key_values is not None:
            k, v, k_head, v_head = past_key_values.update(
                mpa_state=(k, v, k_head, v_head),
                layer_idx=self.layer_idx,
                offset=n,
            )["mpa_state"]

        causal = True if self.training or q.shape[-3] == k.shape[-3] else False
        window_size = (self.window_size, 0) if self.window_size > 0 else (-1, -1)

        # only use cu_seqlens in training
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if (cu_seqlens is not None) or (
            attention_mask is not None and not attention_mask.all()
        ):
            if cu_seqlens is not None:  # flame training state
                cu_seqlens_q = cu_seqlens_k = cu_seqlens
                max_seqlen_q = max_seqlen_k = n
                q = q.squeeze(0)
                k = k.squeeze(0)
                v = v.squeeze(0)
            else:
                q, k, v, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
                    q=q, k=k, v=v, attention_mask=attention_mask, q_len=n
                )
                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_q, max_seqlen_k = max_seq_lens

            output = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=causal,
                window_size=window_size,
            )

            if cu_seqlens is None:
                output = pad_input(output, indices_q, b, n)
            else:
                output = output.unsqueeze(0)
        else:
            output = flash_attn_func(q, k, v, causal=causal, window_size=window_size)

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        # outproj
        output = self.o_proj(output)

        return output, past_key_values
