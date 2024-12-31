import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
from transformers.cache_utils import Cache
from xopes.ops import householder_fn

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import EMBED_DIM_BASE, XMIXERS_DEBUG, print_params


class Hgru3(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expand_ratio: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        norm_type: str = "layernorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        beta_activation: str = "silu",
        causal: bool = True,
        use_dense_memory: bool = True,
        token_mixer_init_type: int = 0,
        rescale_type: int = 0,
        num_layers: int = 12,
        init_std: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        assert causal, f"Only causal={causal} is supported"

        self.layer_idx = layer_idx
        self.expand_ratio = expand_ratio
        self.causal = causal
        self.use_dense_memory = use_dense_memory
        self.use_output_gate = use_output_gate

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.f_proj = nn.Linear(embed_dim, embed_dim // expand_ratio, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.beta_act = get_activation_fn(beta_activation)
        self.norm = get_norm_fn(norm_type)(embed_dim, bias=False)

        if self.use_dense_memory:
            # !!! dont use beta as name in hf: https://github.com/huggingface/transformers/issues/29554
            # I - 2 * beta beta ^ T
            self.bet_proj = nn.Sequential(
                nn.Linear(embed_dim, expand_ratio, bias=bias),
                nn.Linear(expand_ratio, embed_dim, bias=bias),
            )

        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, expand_ratio, bias=bias),
                nn.Linear(expand_ratio, embed_dim, bias=bias),
            )

        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
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
        elif self.token_mixer_init_type == 3:  # minicpm init
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight,
                    gain=self.init_std / ((self.embed_dim / EMBED_DIM_BASE) ** 0.5),
                )
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
        use_cache: Optional[bool] = False,
        log_lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # x: b n d
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        f = self.f_proj(x)

        # act
        q = self.q_act(q)
        k = self.k_act(k)

        if self.use_dense_memory:
            # I - 2 beta beta ^ T
            beta = self.beta_act(self.bet_proj(x))
            q = householder_fn(q, beta)
            # print(q.shape, beta.shape)
            # beta = l2_norm(beta, 1e-6).contiguous() / (beta.shape[-1] ** 0.5)
            # q_beta = (q * beta).sum(dim=-1, keepdim=True)
            # q = q - 2 * q_beta * beta

        # h is num_head, d is head dimension
        q, k, v = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [q, k, v],
        )

        # todo: make a fusion here
        # l + (1 - l) * sigmoid(x)
        lower_bound = torch.exp(log_lower_bound.float())
        f = lower_bound + (1 - lower_bound) * F.sigmoid(f.float())
        log_f = torch.log(f)

        q, k, v, log_f = map(
            lambda x: rearrange(x, "b n h ... -> b h n ...").contiguous(),
            [q, k, v, log_f],
        )

        recurrent_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"]

        if self.causal:
            if self.training or use_cache:
                q = q.to(k.dtype)
                fn = chunk_simple_gla
            else:
                # q = q.to(k.dtype)
                # q = q.float()
                # k = k.float()
                # v = v.float()
                # log_f = log_f.float()
                fn = fused_recurrent_simple_gla

            output, recurrent_state = fn(
                q=q,
                k=k,
                v=v,
                g=log_f,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
            output = output.to(x.dtype)
        else:
            assert False

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx,
                offset=x.shape[-2],
            )

        output = rearrange(output, "b h n d -> b n (h d)")

        if self.use_output_gate:
            output_gate = F.sigmoid(self.output_gate(x))
            output = output * output_gate

        # use post norm here for better parallel when using tp
        output = self.norm(output)

        # out proj
        output = self.out_proj(output)

        return output, past_key_values
