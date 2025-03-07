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
        token_mixer_norm_type: str = "rmsnorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        v_activation: str = "silu",
        use_beta: bool = True,
        beta_activation: str = "neg",
        use_decay: bool = False,
        scalar_decay: bool = False,
        qkv_norm_type: int = 2,
        norm_q: bool = False,
        norm_v: bool = False,
        causal: bool = True,
        max_position_embeddings: int = 1024,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.01,
        gate_act: str = "sigmoid",
        gate_pos: str = "pre",
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
        self.use_beta = use_beta
        if self.use_beta:
            self.beta_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.v_act = get_activation_fn(v_activation)
        self.beta_activation = beta_activation
        norm_type = (
            f"{token_mixer_norm_type}_fused_gate"
            if use_output_gate
            else token_mixer_norm_type
        )
        self.norm = get_norm_fn(norm_type)(
            embed_dim,
            bias=bias,
            gate_act=gate_act,
            gate_pos=gate_pos,
            num_groups=num_heads,
        )

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
        self.beta = torch.empty(0)
        self.init_state = torch.empty(0)

    def _init_weights(self):
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        return _initialize_weights(self, module)

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
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
            # l + (1 - l) * sigmoid(x)
            if lower_bound is not None:
                f = F.sigmoid(self.f_proj(x))
                f = lower_bound + (1 - lower_bound) * f
                log_f = torch.log(f)
            else:
                log_f = F.logsigmoid(self.f_proj(x))
        else:
            log_f = None
        # b n h
        if self.use_beta:
            beta = F.sigmoid(self.beta_proj(x))
            if self.beta_activation == "neg":
                beta = beta * 2
        else:
            # if no beta, use 2 to get householder matrix
            if self.beta.shape[0] == 0 or self.beta.shape != torch.Size([b, n, h]):
                self.beta = torch.ones(b, n, h).to(q) * 2
            beta = self.beta
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
        q_offset = 0
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )
        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1).unsqueeze(-1)
            k = k.masked_fill(attention_mask_ == 0, 0)
            if log_f is not None:
                log_f = log_f.masked_fill(attention_mask_ == 0, 0)

        scale = 1
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
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=scale,
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
                    a=(k * -beta.unsqueeze(-1) * torch.exp(log_f)).to(dtype),
                    b=k.to(dtype),
                    gk=log_f.to(dtype),
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=scale,
                    head_first=False,
                )
        else:
            assert False

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx,
                offset=n,
            )

        # reshape
        output = rearrange(output, "b n h d -> b n (h d)")
        if self.use_output_gate:
            gate = self.output_gate(x)
            output = self.norm(output, gate)
        else:
            output = self.norm(output)

        # out proj
        output = self.o_proj(output)

        return output, past_key_values
