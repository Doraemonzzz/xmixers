# multi product attention
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.utils import XMIXERS_DEBUG, print_module, print_params

from ...pes import Lrpe

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None


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
        token_mixer_init_type: int = 0,
        rescale_type: int = 0,
        num_layers: int = 12,
        mpa_type: int = 0,
        mpa_activation: str = "none",
        head_dim: int = -1,
        gate_type: int = 0,
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
        self.gate_type = gate_type
        if self.gate_type == 1:
            self.gate_proj = nn.Linear(embed_dim, self.num_heads * 2, bias=bias)
        elif self.gate_type == 2:
            self.gate_proj = nn.Linear(embed_dim, self.head_dim * 2, bias=bias)

        self.act = get_activation_fn(mpa_activation)

        self.out_proj = nn.Linear(mid_dim, embed_dim, bias=bias)

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(
                head_dim=self.head_dim,
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
            if hasattr(module, "k_head"):
                nn.init.xavier_uniform_(module.k_head, gain=2**-2.5)
            if hasattr(module, "v_head"):
                nn.init.xavier_uniform_(module.v_head, gain=2**-2.5)
        elif self.token_mixer_init_type == 2:  # fairseq init
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=2**-0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if hasattr(module, "k_head"):
                nn.init.xavier_uniform_(module.k_head, gain=2**-0.5)
            if hasattr(module, "v_head"):
                nn.init.xavier_uniform_(module.v_head, gain=2**-0.5)

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
        b = x.shape[0]
        # x: b n d
        # linear map
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        if self.mpa_type == 0:
            kv_head = self.act(self.kv_head_proj(x))
        else:
            kv_head = self.act(self.kv_head)

        # TODO: add a kernel here
        if self.gate_type == 1:
            # b n h
            kv_head_gate = self.gate_proj(x).float()
            zero = torch.zeros(
                [b, 1, self.num_heads],
                device=kv_head_gate.device,
                dtype=kv_head_gate.dtype,
            )
            log_f = torch.cat([zero, F.logsigmoid(kv_head_gate)], dim=1)
            log_f_cumsum = torch.cumsum(log_f, dim=-2)[:, :-1]
            kv_head = kv_head * torch.exp(log_f_cumsum)
            kv_head = torch.cumsum(kv_head, dim=-2).to(kv_head.dtype)
        elif self.gate_type == 2:
            # b n d
            kv_gate = self.gate_proj(x)
            zero = torch.zeros(
                [b, 1, self.head_dim], device=kv_gate.device, dtype=kv_gate.dtype
            )
            log_f = torch.cat([zero, F.logsigmoid(kv_gate)], dim=1)
            log_f_cumsum = torch.cumsum(log_f, dim=-2)[:, :-1]
            kv = (kv * torch.exp(log_f_cumsum)).to(kv_head.dtype)
            kv = torch.cumsum(kv, dim=-2).to(kv_head.dtype)

        # for lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        # TODO: upate cache update
        if past_key_values is not None:
            k, v, k_head, v_head = past_key_values.update(
                mpa_state=(k, v, k_head, v_head),
                layer_idx=self.layer_idx,
                offset=x.shape[-2],
            )["mpa_state"]

        # construct k, v cache
        # todo: add a fusion here
        k, v = map(
            lambda arr: torch.einsum("... d, ... h -> ... h d", arr[0], arr[1]),
            [(k, k_head), (v, v_head)],
        )
        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads)

        q, k, v = map(
            lambda x: rearrange(x, "... n h d -> ... h n d"),
            [q, k, v],
        )

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k)

        # if (
        #     attention_mask is None or attention_mask.all()
        # ):  # if attention mask is None or all elements are True, use sdpa
        if True:
            # use causal when training or evaluation(not for generation) or prefill
            is_causal = True if self.training or q.shape[-2] == k.shape[-2] else False
            # output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            q, k, v = map(lambda x: rearrange(x, "... h n d -> ... n h d"), [q, k, v])
            dtype = q.dtype
            if dtype == torch.float32:
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)
            output = flash_attn_func(q, k, v, causal=is_causal).to(dtype)
            output = rearrange(output, "... n h d -> ... h n d")
        else:
            assert False, "flash_attn_varlen_qkvpacked_func current not support"

        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")
        # outproj
        output = self.out_proj(output)

        return output, past_key_values
