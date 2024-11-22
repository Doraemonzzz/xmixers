# Tnl: https://arxiv.org/pdf/2405.17381
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, print_params

from ...pes import Lrpe


class TnlAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_heads: int = -1,
        bias: bool = False,
        use_lrpe: bool = False,
        layer_idx: int = 0,
        lrpe_type: int = 1,
        base: int = 10000,
        gate_dim: int = 16,
        use_output_gate: bool = True,
        norm_type: str = "layernorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        v_activation: str = "silu",
        causal: bool = True,
        norm_pos: str = "post",  # choose from ["attn", "ogate", "post"]
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
        self.norm = get_norm_fn(norm_type)(embed_dim)
        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.v_act = get_activation_fn(v_activation)
        self.causal = causal
        self.norm_pos = norm_pos

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                lrpe_type=lrpe_type,
                base=base,
                max_position_embeddings=max_position_embeddings,
            )

        self.use_output_gate = use_output_gate
        if self.use_output_gate:
            self.out_gate = nn.Sequential(
                nn.Linear(embed_dim, gate_dim, bias=bias),
                nn.Linear(gate_dim, embed_dim, bias=bias),
            )

        self.causal_mask = None
        self.max_position_embeddings = max_position_embeddings
        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if getattr(module, "_is_hf_initialized", False):
            return

        if self.token_mixer_init_type == 0:
            pass
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
        log_slope: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        # x: b n d
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # act
        q = self.q_act(q)
        k = self.k_act(k)
        v = self.v_act(v)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", d=self.head_dim),
            [q, k, v],
        )

        # lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        # cache update
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k)

        if self.causal:
            if self.causal_mask is None:
                index = torch.arange(
                    self.max_position_embeddings, dtype=torch.int64, device=x.device
                ).float()
                mask = index.unsqueeze(1) - index.unsqueeze(0)
                self.causal_mask = torch.exp(
                    log_slope.unsqueeze(-1).unsqueeze(-1) * mask
                ).masked_fill(mask < 0, 0)
            energy = torch.einsum("... n d, ... m d -> ... n m", q, k)
            # use causal when training or evaluation(not for generation) or prefill
            is_causal = True if self.training or (q.shape[-2] == k.shape[-2]) else False
            if is_causal:
                n = k.shape[-2]
                causal_mask = self.causal_mask[:n, :n]
                energy = energy * causal_mask
            output = torch.einsum("... n m, ... m d -> ... n d", energy, v)
        else:
            assert False, "not implemented"

        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")

        if self.norm_pos == "attn":
            output = self.norm(output)

        if self.use_output_gate:
            output_gate = F.sigmoid(self.out_gate(x))
            output = output * output_gate

        if self.norm_pos == "ogate":
            output = self.norm(output)

        # outproj
        output = self.out_proj(output)

        if self.norm_pos == "post":
            # use post norm here for better parallel when using tp
            output = self.norm(output)

        return output, past_key_values
