# flex attention: https://pytorch.org/blog/flexattention/
import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from xmixers.utils import XMIXERS_DEBUG, print_params

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None


class FlexAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_heads: int = -1,
        bias: bool = False,
        layer_idx: int = 0,
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

        self.layer_idx = layer_idx
        self.kv_heads = kv_heads
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.kv_heads == -1:
            kv_dim = embed_dim
        else:
            kv_dim = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

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
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        log_slope: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # x: b n d
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v],
        )

        # lrpe
        if past_key_values is not None:
            past_key_values.get_seq_length(self.layer_idx)

        # cache update
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        if (
            attention_mask is None or attention_mask.all()
        ):  # if attention mask is None or all elements are True, use sdpa
            # use causal when training or evaluation(not for generation) or prefill
            is_causal = True if self.training or q.shape[-2] == k.shape[-2] else False
            output = flash_attn_func(q, k, v, causal=is_causal, alibi_slopes=log_slope)
        else:
            assert False, "flash_attn_varlen_qkvpacked_func current not support"

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        # outproj
        output = self.out_proj(output)

        return output, past_key_values
