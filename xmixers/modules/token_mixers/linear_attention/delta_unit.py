from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule
from fla.ops.generalized_delta_rule import (
    chunk_dplr_delta_rule,
    fused_recurrent_dplr_delta_rule,
)
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn, l2_norm
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params


class DeltaUnit(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        norm_type: str = "layernorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        v_activation: str = "silu",
        use_gamma: bool = True,
        gamma_activation: str = "pos",
        use_decay: bool = True,
        scalar_decay: bool = True,
        qkv_norm_type: int = 2,
        norm_q: bool = False,
        norm_v: bool = False,
        causal: bool = True,
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
        self.causal = causal
        self.use_output_gate = use_output_gate

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.use_gamma = use_gamma
        if self.use_gamma:
            self.gamma_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.v_act = get_activation_fn(v_activation)
        self.gamma_activation = gamma_activation
        self.norm = get_norm_fn(norm_type)(embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )
        self.use_decay = use_decay
        self.scalar_decay = scalar_decay
        if self.use_decay:
            if self.scalar_decay:
                self.f_proj = nn.Linear(embed_dim, num_heads, bias=bias)
            else:
                self.f_proj = nn.Sequential(
                    nn.Linear(embed_dim, self.head_dim, bias=bias),
                    nn.Linear(self.head_dim, embed_dim, bias=bias),
                )

        self.qkv_norm_type = qkv_norm_type
        self.norm_q = norm_q
        self.norm_v = norm_v
        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_std = init_std
        self.gain = gain
        self.apply(self._initialize_weights)
        self.f = torch.empty(0)
        self.zero = torch.empty(0)
        self.gamma = torch.empty(0)
        self.init_state = torch.empty(0)

    def _initialize_weights(self, module):
        return _initialize_weights(self, module)

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        b, n, d = x.shape
        h = self.num_heads
        # x: b n d
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # b n h
        if self.use_decay:
            log_f = F.logsigmoid(self.f_proj(x))
        # b n h
        if self.use_gamma:
            gamma = F.sigmoid(self.gamma_proj(x))
            if self.gamma_activation == "neg":
                gamma = gamma * 2
        else:
            if self.gamma.shape[0] == 0 or self.gamma.shape != torch.Size([b, n, h]):
                self.gamma = torch.ones(b, n, h).to(q) * 2
            gamma = self.gamma
        gamma = -gamma
        # act
        q = self.q_act(q)
        k = self.k_act(k)

        # h is num_head, d is head dimension
        q, k, v = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim),
            [q, k, v],
        )

        if self.norm_q:
            q = l2_norm(q)
        k = l2_norm(k)
        if self.norm_v:
            v = l2_norm(v)

        if self.use_decay:
            if self.scalar_decay:
                log_f = repeat(log_f, "... h -> ... h d", d=self.head_dim)
            else:
                log_f = rearrange(log_f, "... (h d) -> ... h d", d=self.head_dim)

        recurrent_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"]

        if self.causal:
            dtype = q.dtype
            if not self.use_decay:
                if self.training or use_cache:
                    fn = chunk_delta_rule
                else:
                    fn = fused_recurrent_delta_rule

                output, recurrent_state = fn(
                    q=q,
                    k=k.to(dtype),
                    v=v.to(dtype),
                    beta=-gamma,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=1,
                    head_first=False,
                )
            else:
                if self.training or use_cache:
                    fn = chunk_dplr_delta_rule
                else:
                    fn = fused_recurrent_dplr_delta_rule

                output, recurrent_state = fn(
                    q=q,
                    k=k.to(dtype),
                    v=v.to(dtype),
                    a=(k * gamma.unsqueeze(-1) * torch.exp(log_f)).to(dtype),
                    b=k.to(dtype),
                    gk=log_f.to(dtype),
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=1,
                    head_first=False,
                )
        else:
            assert False

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx,
                offset=x.shape[-2],
            )

        # reshape
        output = rearrange(output, "b n h d -> b n (h d)")

        if self.use_output_gate:
            output_gate = F.sigmoid(self.output_gate(x))
            output = output * output_gate

        # use post norm here for better parallel when using tp
        output = self.norm(output)

        # out proj
        output = self.out_proj(output)

        return output, past_key_values
