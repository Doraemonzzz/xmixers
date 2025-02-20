# adapt from https://github.com/deepseek-ai/DeepSeek-V2

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.cache_utils import Cache

from xmixers.modules.pes import Lrpe
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params

from .utils import _upad_input

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except:
    flash_attn_func = None
    index_first_axis = None
    pad_input = None
    unpad_input = None


class MultiLatentAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_lora_rank: int = 512,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        bias: bool = False,
        use_lrpe: bool = True,
        layer_idx: int = 0,
        lrpe_type: int = 1,
        base: int = 10000,
        max_position_embeddings: int = 1024,
        token_mixer_init_type: int = 0,
        rescale_type: int = 0,
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

        qk_nope_head_dim = embed_dim // num_heads
        v_head_dim = embed_dim // num_heads
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.window_size = window_size

        # d -> r1
        self.q_a_proj = nn.Linear(embed_dim, q_lora_rank, bias=bias)
        # r1 -> h * (e_nope + e_pe)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.q_head_dim, bias=bias)

        # d -> r2 + e_pe
        self.kv_a_proj_with_mqa = nn.Linear(
            embed_dim,
            kv_lora_rank + qk_rope_head_dim,
            bias=bias,
        )
        # r2 -> h * (e_nope + e_v)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=bias,
        )

        # h * e_v -> d
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            embed_dim,
            bias=bias,
        )

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(
                head_dim=qk_rope_head_dim,
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
        self._init_weights()

    def _init_weights(self):
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        return _initialize_weights(self, module)

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        b, n, d = x.shape
        # x: b n d
        q = self.q_b_proj(self.q_a_proj(x))
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = repeat(k_pe, "b n d -> b n h d", h=self.num_heads)

        kv = self.kv_b_proj(compressed_kv)
        kv = rearrange(kv, "b n (h d) -> b n h d", h=self.num_heads)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # for lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        if self.use_lrpe:
            q_pe = self.lrpe(q_pe, offset=q_offset)
            k_pe = self.lrpe(k_pe, offset=q_offset)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        e = v.shape[-1]
        if k.shape[-1] != e:
            v = F.pad(v, (0, k.shape[-1] - v.shape[-1]))

        # cache update
        if past_key_values is not None:
            k, v = past_key_values.update(
                attn_state=(k, v),
                layer_idx=self.layer_idx,
                offset=n,
            )["attn_state"]

        causal = True if self.training or q.shape[-3] == k.shape[-3] else False
        window_size = (self.window_size, 0) if self.window_size > 0 else (-1, -1)

        # only use cu_seqlens in training
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if (cu_seqlens is not None) or (
            attention_mask is not None and not attention_mask.all()
        ):
            if cu_seqlens is not None:  # flame training stage
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

        output = output[:, :, :, :e]

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        # outproj
        output = self.o_proj(output)

        return output, past_key_values
