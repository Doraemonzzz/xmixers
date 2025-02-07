import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
from transformers.cache_utils import Cache
from xopes.ops import cumsum_fn

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params


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
        use_dense_memory: bool = False,
        scalar_decay: bool = False,
        token_mixer_init_type: int = 0,
        rescale_type: int = 0,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.02,
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
        self.scalar_decay = scalar_decay
        if self.scalar_decay:
            self.f_proj = nn.Linear(embed_dim, embed_dim // expand_ratio, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.norm = get_norm_fn(norm_type)(embed_dim, bias=bias)

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
        self.gain = gain
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        return _initialize_weights(self, module)

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
        if self.scalar_decay:
            f = F.sigmoid(self.f_proj(x))
            k = self.k_act(k)
        else:
            k = F.sigmoid(k)
            f = 1 - k

        # act
        q = self.q_act(q)

        # todo: make a fusion here
        # l + (1 - l) * sigmoid(x)
        if log_lower_bound is not None:
            lower_bound = torch.exp(log_lower_bound)
            f = lower_bound + (1 - lower_bound) * f
        log_f = torch.log(f)

        # h is num_head, d is head dimension
        q, k, v = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [q, k, v],
        )
        if not self.scalar_decay:
            f = rearrange(f, "... (h d) -> ... h d", d=self.expand_ratio)

        sign_f = torch.sign(f - 0.5)
        if self.scalar_decay:
            sign_f = sign_f.unsqueeze(-1)
        theta = cumsum_fn(sign_f, dim=1) * math.pi
        sin = torch.sin(theta)
        cos = torch.cos(theta)
        q = torch.cat([q * cos, q * sin], dim=-1)
        k = torch.cat([k * cos, k * sin], dim=-1)
        if not self.scalar_decay:
            log_f = torch.cat([log_f, log_f], dim=-1)

        recurrent_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"]

        if self.causal:
            dtype = q.dtype
            if self.scalar_decay:
                if self.training or use_cache:
                    fn = chunk_simple_gla
                else:
                    fn = fused_recurrent_simple_gla
            else:
                if self.training or use_cache:
                    fn = chunk_gla
                else:
                    fn = fused_recurrent_gla

            output, recurrent_state = fn(
                q=q,
                k=k.to(dtype),
                v=v.to(dtype),
                g=log_f.to(dtype),
                initial_state=recurrent_state,
                output_final_state=use_cache,
                head_first=False,
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

        output = rearrange(output, "... h d -> ... (h d)")

        if self.use_output_gate:
            output_gate = F.sigmoid(self.output_gate(x))
            output = output * output_gate

        # use post norm here for better parallel when using tp
        output = self.norm(output)

        # out proj
        output = self.out_proj(output)

        return output, past_key_values
