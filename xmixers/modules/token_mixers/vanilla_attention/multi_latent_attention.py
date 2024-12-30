# adapt from https://github.com/deepseek-ai/DeepSeek-V2

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.cache_utils import Cache

from xmixers.utils import XMIXERS_DEBUG, print_params

from ...pes import Lrpe

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None


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
        self.out_proj = nn.Linear(
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
                max_position_embeddings=max_position_embeddings,
            )

        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if getattr(module, "_is_hf_initialized", False):
            return

        if self.token_mixer_init_type == 0:
            return
        elif self.token_mixer_init_type == 1:  # fla init
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif self.token_mixer_init_type == 2:  # fairseq init
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=2**-0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if self.rescale_type == 1:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference: https://github.com/karpathy/nanoGPT/blob/master/model.py#L144 https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/models/gla/modeling_gla.py#L152
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    num_residuals_per_layer = 2
                    # module.weight.data.normal_(mean=0.0, std=std/math.sqrt(2 * self.config.num_layers))
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.num_layers)

        module._is_hf_initialized = True

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # x: b n d
        q = self.q_b_proj(self.q_a_proj(x))
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = repeat(k_pe, "b n d -> b n h d", h=self.num_heads)
        k_pe = rearrange(k_pe, "b n h d -> b h n d")

        kv = self.kv_b_proj(compressed_kv)
        kv = rearrange(kv, "b n (h d) -> b h n d", h=self.num_heads)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # for lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        # TODO: cache update

        if self.use_lrpe:
            q_pe = self.lrpe(q_pe, offset=q_offset)
            k_pe = self.lrpe(k_pe)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        if (
            attention_mask is None or attention_mask.all()
        ):  # if attention mask is None or all elements are True, use sdpa
            # use causal when training or evaluation(not for generation) or prefill
            is_causal = True if self.training or q.shape[-2] == k.shape[-2] else False
            output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        else:
            assert False, "flash_attn_varlen_qkvpacked_func current not support"

        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")
        # outproj
        output = self.out_proj(output)

        return output, past_key_values
