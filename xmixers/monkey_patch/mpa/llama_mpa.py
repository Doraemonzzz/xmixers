from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    apply_rotary_pos_emb,
)

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None


class LlamaMpaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.kv_proj = nn.Linear(
            config.hidden_size, 2 * self.head_dim, bias=config.attention_bias
        )
        self.kv_head_proj = nn.Linear(
            config.hidden_size,
            2 * config.num_attention_heads,
            bias=config.attention_bias,
        )
        self.act = F.sigmoid
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # x: b n d
        # linear map
        q = self.q_proj(hidden_states)
        k, v = self.kv_proj(hidden_states).chunk(2, dim=-1)
        k_head, v_head = self.kv_head_proj(hidden_states).chunk(2, dim=-1)

        k_head = self.act(k_head)
        v_head = self.act(v_head)

        k, v = map(
            lambda arr: torch.einsum("... d, ... h -> ... h d", arr[0], arr[1]),
            [(k, k_head), (v, v_head)],
        )
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)

        query_states, key_states, value_states = map(
            lambda x: rearrange(x, "... n h d -> ... h n d"),
            [q, k, v],
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = None
        query_states, key_states, value_states = map(
            lambda x: rearrange(x.to(value_states.dtype), "b h n d -> b n h d"),
            [query_states, key_states, value_states],
        )
        attn_output = flash_attn_func(
            query_states, key_states, value_states, softmax_scale=self.scaling
        )
        attn_output = rearrange(attn_output, "b n h d -> b n (h d)")

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class LlamaMpaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = LlamaMpaAttention(config, layer_idx)
