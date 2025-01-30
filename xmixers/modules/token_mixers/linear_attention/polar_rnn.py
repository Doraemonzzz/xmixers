from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.generalized_delta_rule import (
    chunk_dplr_delta_rule,
    fused_recurrent_dplr_delta_rule,
)
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params


class PolarRnn(nn.Module):
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
        scaler_decay: bool = True,
        qkv_norm_type: int = 2,
        norm_q: bool = False,
        norm_v: bool = False,
        causal: bool = True,
        token_mixer_init_type: int = 0,
        rescale_type: int = 0,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.02,
        debug: int = 0,
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
        self.use_decay = use_decay
        self.scaler_decay = scaler_decay
        if self.scaler_decay:
            self.f_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        self.use_gamma = use_gamma
        if self.use_gamma:
            self.gamma_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.v_act = get_activation_fn(v_activation)
        self.gamma_activation = gamma_activation
        self.norm = get_norm_fn(norm_type)(embed_dim, bias=False)

        self.head_dim = embed_dim // num_heads
        if self.use_output_gate:
            self.output_gate = nn.Sequential(
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
        self.debug = debug  # rm later

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

        if self.zero.shape[0] == 0:
            self.zero = torch.zeros(b, n, h, d // h).to(q)
        # b n h
        if self.use_decay:
            if self.scaler_decay:
                f = F.logsigmoid(self.f_proj(x))
                v = self.v_act(v)
            else:
                f = F.logsigmoid(-v)
                v = F.sigmoid(v)
        else:
            if self.f.shape[0] == 0:
                self.f = torch.zeros(b, n, h).to(q)
            f = self.f
            v = self.v_act(v)
        # b n h
        if self.use_gamma:
            gamma = F.sigmoid(self.gamma_proj(x))
            if self.gamma_activation == "neg":
                gamma *= 2
        else:
            if self.gamma.shape[0] == 0:
                self.gamma = torch.ones(b, n, h).to(q) * 2
            gamma = self.gamma
        # act
        q = self.q_act(q)
        k = self.k_act(k)

        # h is num_head, d is head dimension
        q, k, v = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim),
            [q, k, v],
        )

        if self.norm_q:
            q = F.normalize(q, p=self.qkv_norm_type, dim=-1)
        k = F.normalize(k, p=self.qkv_norm_type, dim=-1)
        if self.norm_v:
            v = F.normalize(v, p=self.qkv_norm_type, dim=-1)

        if self.use_decay and not self.scaler_decay:
            f = rearrange(f, "... (h d) -> ... h d", d=self.head_dim)

        unitary_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            unitary_state = past_key_values[self.layer_idx]["unitary_state"]

        spectral_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            spectral_state = past_key_values[self.layer_idx]["spectral_state"]

        if self.causal:
            if self.debug == 1:
                dtype = q.dtype

                if len(f.shape) == 4:
                    if self.training or use_cache:
                        fn = chunk_gla
                    else:
                        fn = fused_recurrent_gla
                else:
                    if self.training or use_cache:
                        fn = chunk_simple_gla
                    else:
                        fn = fused_recurrent_simple_gla

                output, spectral_state = fn(
                    q=q.to(dtype),
                    k=k.to(dtype),
                    v=v.to(dtype),
                    g=f.to(dtype),
                    initial_state=spectral_state,
                    output_final_state=use_cache,
                    scale=1,
                    head_first=False,
                )
            elif self.debug == 2:
                # Unitary update
                if self.training or use_cache:
                    fn = chunk_dplr_delta_rule
                else:
                    fn = fused_recurrent_dplr_delta_rule

                dtype = q.dtype

                output, unitary_state = fn(
                    q=q,
                    k=k.to(dtype),
                    v=v.to(dtype),
                    a=(k * gamma.unsqueeze(-1)).to(dtype),
                    b=k.to(dtype),
                    gk=self.zero.to(dtype),
                    initial_state=unitary_state,
                    output_final_state=use_cache,
                    scale=1,
                    head_first=False,
                )
            else:
                # Unitary update
                if self.training or use_cache:
                    fn = chunk_dplr_delta_rule
                else:
                    fn = fused_recurrent_dplr_delta_rule

                dtype = q.dtype

                output, unitary_state = fn(
                    q=q,
                    k=self.zero.to(dtype),
                    v=self.zero.to(dtype),
                    a=(k * gamma.unsqueeze(-1)).to(dtype),
                    b=k.to(dtype),
                    gk=self.zero.to(dtype),
                    initial_state=unitary_state,
                    output_final_state=use_cache,
                    scale=1,
                    head_first=False,
                )

                # Spectral update
                if len(f.shape) == 4:
                    if self.training or use_cache:
                        fn = chunk_gla
                    else:
                        fn = fused_recurrent_gla
                else:
                    if self.training or use_cache:
                        fn = chunk_simple_gla
                    else:
                        fn = fused_recurrent_simple_gla

                output, spectral_state = fn(
                    q=output.to(dtype),
                    k=v.to(dtype),
                    v=v.to(dtype),
                    g=f.to(dtype),
                    initial_state=spectral_state,
                    output_final_state=use_cache,
                    scale=1,
                    head_first=False,
                )
        else:
            assert False

        if past_key_values is not None:
            past_key_values.update(
                unitary_state=unitary_state,
                spectral_state=spectral_state,
                layer_idx=self.layer_idx,
                offset=x.shape[-2],
            )

        output = rearrange(output, "b n h d -> b n (h d)")

        if self.use_output_gate:
            output_gate = F.sigmoid(self.output_gate(x))
            output = output * output_gate

        # use post norm here for better parallel when using tp
        output = self.norm(output)

        # out proj
        output = self.out_proj(output)

        return output, past_key_values
