import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import EMBED_DIM_BASE, XMIXERS_DEBUG, print_params


class MetaLa(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expand_ratio: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        non_sparse_ratio: float = 1,
        num_sparse: int = 4,
        norm_type: str = "layernorm",
        q_activation: str = "silu",
        causal: bool = True,
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
        self.use_output_gate = use_output_gate

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        self.norm = get_norm_fn(norm_type)(embed_dim, bias=bias)

        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, expand_ratio, bias=bias),
                nn.Linear(expand_ratio, embed_dim, bias=bias),
            )

        self.non_sparse_ratio = non_sparse_ratio
        self.num_sparse = num_sparse
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
        elif self.token_mixer_init_type == 2:  # fairseq init
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=2**-0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
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
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # x: b n d
        # linear map
        # !!! not q, log_f, v, this is for align with internel version
        v, q, log_f = self.in_proj(x).chunk(3, dim=-1)

        # act
        q = self.q_act(q)
        f = F.sigmoid(log_f)

        # todo: make a fusion here
        # l + (1 - l) * sigmoid(x)
        if lower_bound is not None:
            f = lower_bound + (1 - lower_bound) * f
            log_f = torch.log(f)

        q, v, log_f = map(
            lambda x: rearrange(
                x, "b n (h d) -> b h n d", d=self.expand_ratio
            ).contiguous(),
            [q, v, log_f],
        )

        if self.non_sparse_ratio < 1:
            num_non_sparse = int(log_f.shape[-1] * self.non_sparse_ratio)
            num_sparse = log_f.shape[-1] - num_non_sparse
            # non sparse
            log_f_non_sparse, log_f_sparse = log_f.split(
                [num_non_sparse, num_sparse], dim=-1
            )
            k_non_sparse = 1 - torch.exp(log_f_non_sparse)
            # sparse
            k_sparse = F.softmax(1 - log_f_sparse, dim=-1)
            # find topk smallest values and set to 0
            _, index = torch.topk(
                k_sparse,
                num_sparse - self.num_sparse,
                dim=-1,
                largest=False,
                sorted=False,
                out=None,
            )
            k_sparse = k_sparse.scatter(
                -1, index, torch.zeros_like(k_sparse, device=k_sparse.device)
            )
            log_f_sparse = (1 - k_sparse).log()

            # concat
            log_f = torch.cat([log_f_non_sparse, log_f_sparse], dim=-1)
            k = torch.cat([k_non_sparse, k_sparse], dim=-1)
        else:
            k = 1 - f

        recurrent_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"]

        dtype = q.dtype
        q, k, v, log_f = map(lambda x: x.to(dtype), [q, k, v, log_f])
        if self.causal:
            if self.training or recurrent_state is None:  # training or prefilling
                fn = chunk_gla
            else:
                fn = fused_recurrent_gla

            output, recurrent_state = fn(
                q=q,
                k=k,
                v=v,
                g=log_f,
                scale=1,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
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
