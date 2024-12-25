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


class LinearAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_heads: int = -1,
        bias: bool = False,
        use_lrpe: bool = True,
        layer_idx: int = 0,
        lrpe_type: int = 1,
        base: int = 10000,
        use_output_gate: bool = True,
        norm_type: str = "layernorm",
        linear_activation: str = "silu",
        causal: bool = True,
        use_dense_memory: bool = False,
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
        self.act = get_activation_fn(linear_activation)
        self.causal = causal

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
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )

        self.use_dense_memory = use_dense_memory
        if self.use_dense_memory:
            self.alpha_proj = nn.Linear(embed_dim, 1, bias=bias)
            self.beta_proj = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
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
        q = self.act(q)
        k = self.act(k)

        if self.use_dense_memory:
            alpha = F.sigmoid(self.alpha_proj(x))
            beta = F.normalize(self.beta_proj(x), p=2, dim=-1)
            # q * (I - alpha * beta * beta ^ T)
            q_beta = (q * beta).sum(dim=-1, keepdim=True)
            q = q - alpha * q_beta * beta

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
                self.causal_mask = (
                    torch.tril(
                        torch.ones(
                            self.max_position_embeddings, self.max_position_embeddings
                        )
                    )
                ).to(q)
            energy = torch.einsum("... n d, ... m d -> ... n m", q, k)
            # use causal when training or evaluation(not for generation) or prefill
            is_causal = True if self.training or (q.shape[-2] == k.shape[-2]) else False
            if is_causal:
                n = k.shape[-2]
                causal_mask = self.causal_mask[:n, :n]
                energy = energy * causal_mask
            output = torch.einsum("... n m, ... m d -> ... n d", energy, v)
        else:
            kv = torch.einsum("... h n d, ... h n e -> ... h d e", k, v)
            output = torch.einsum("... h n d, ... h d e -> ... h n e", q, kv)

        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")

        if self.use_output_gate:
            output_gate = F.sigmoid(self.output_gate(x))
            output = output * output_gate

        # use post norm here for better parallel when using tp
        output = self.norm(output)

        # outproj
        output = self.out_proj(output)

        return output, past_key_values
