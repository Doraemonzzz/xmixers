import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.generalized_delta_rule import (
    chunk_dplr_delta_rule,
    fused_recurrent_dplr_delta_rule,
)
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn, l2_norm
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params


class DenseRnn(nn.Module):
    """
    S[t] = (I - k * k ^ T) * Diag(f) * (I - k * k ^ T) * S[t-1] + k * v ^ T
    this is equivalent to the following:
    S[t, 1] = (Diag(f) - Diag(f) * k * k ^ T) * S[t-1] + 0 * 0 ^ T
    S[t, 2] = (I - k * k ^ T) * S[t, 1] + k * v ^ T
    Since dplr has the following form:
        S[t] = (D + a b ^ T) * S[t-1] + k * v ^ T
    We can use the following form:
        D = concat([f, 0])
        a = concat([f * k, k])
        b = concat([k, k])
        k = concat([0, k])
        v = concat([0, v])
        q = concat([0, q])
    to implement.

    Old versioin(deprecated):
    this is equivalent to the following:
    S[t, 1] = (I - k * k ^ T) * S[t-1] + 0 * 0 ^ T
    S[t, 2] = (Diag(f) - 0 * 0 ^ T) * S[t, 1] + 0 * 0 ^ T
    S[t, 3] = (I - k * k ^ T) * S[t, 2] + k * v ^ T
    Since dplr has the following form:
        S[t] = (D + a b ^ T) * S[t-1] + k * v ^ T
    We can use the following form:
        D = concat([0, f, 0])
        a = concat([k, 0, k])
        b = a
        k = concat([0, 0, k])
        v = concat([0, 0, v])
        q = concat([0, 0, q])
    to implement.
    """

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
        threshold: float = 0.99,
        use_bias: bool = False,
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
        self.head_dim = embed_dim // num_heads
        self.f_proj = nn.Sequential(
            nn.Linear(embed_dim, self.head_dim, bias=bias),
            nn.Linear(self.head_dim, embed_dim, bias=bias),
        )
        self.use_beta = use_beta
        if self.use_beta:
            # !!! dont use beta as name in hf: https://github.com/huggingface/transformers/issues/29554
            self.bet_proj = nn.Linear(embed_dim, num_heads, bias=bias)
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

        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )

        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.threshold = threshold

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

    def _init_weights(self):
        self.setup_decay()
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        self.setup_decay()
        return _initialize_weights(self, module)

    def setup_decay(self):
        if not self.use_bias:
            return
        # take x = 0 as median, 1 / (1 + exp(-(median + delta))) = a => 1 + exp(-delta) = 1 / a => exp(-delta) = (1 / a - 1) -> exp(delta) = a / (1 - a) => delta = log(a / (1 - a))
        a = self.threshold
        bias = torch.ones(self.embed_dim) * math.log(a / (1 - a))
        if hasattr(self, "bias"):
            if isinstance(self.bias, DTensor):
                self.bias.data.copy_(
                    DTensor.from_local(
                        bias,
                        device_mesh=self.bias.device_mesh,
                    )
                )
            else:
                self.bias.data.copy_(bias)
        else:
            self.bias = nn.Parameter(bias, requires_grad=True)

    def extra_repr(self):
        return print_module(self)

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
        h = d // self.head_dim
        # x: b n d
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.use_bias:
            f = self.f_proj(x) + self.bias
        else:
            f = self.f_proj(x)

        if self.zero.shape[0] == 0 or self.zero.shape[1] != n:
            self.zero = torch.zeros(b, n, h, d // h).to(q)
        # l + (1 - l) * sigmoid(x)
        if lower_bound is not None:
            f = lower_bound + (1 - lower_bound) * F.sigmoid(f)
            log_f = torch.log(f)
        else:
            log_f = F.logsigmoid(f)
        # b n h 1
        if self.use_beta:
            beta = F.sigmoid(self.bet_proj(x)).unsqueeze(-1)
            if self.beta_activation == "neg":
                beta = beta * 2
        else:
            beta = 2
        # act
        q = self.q_act(q)
        k = self.k_act(k)
        v = self.v_act(v)

        # h is num_head, d is head dimension
        q, k, v, log_f = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim),
            [q, k, v, log_f],
        )

        if self.norm_q:
            q = l2_norm(q)
        k = l2_norm(k)
        if self.norm_v:
            v = l2_norm(v)

        # TODO: update this
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
            log_f = log_f.masked_fill(attention_mask_ == 0, 0)

        # old version
        # D = concat([0, f, 0])
        # a = concat([k, 0, k])
        # b = a
        # k = concat([0, 0, k])
        # v = concat([0, 0, v])
        # q = concat([0, 0, q])
        # k_ = k * gamma
        # log_f = torch.cat([self.zero, log_f, self.zero], dim=1)
        # a = torch.cat([k_, self.zero, k_], dim=1)
        # b = torch.cat([k, self.zero, k], dim=1)
        # k = torch.cat([self.zero, self.zero, k], dim=1)
        # v = torch.cat([self.zero, self.zero, v], dim=1)
        # q = torch.cat([self.zero, self.zero, q], dim=1)

        # D = concat([f, 0])
        # a = concat([f * k, k])
        # b = concat([k, k])
        # k = concat([0, k])
        # v = concat([0, v])
        # q = concat([0, q])
        # log_f, a, b, k, v, q = map(
        #     lambda x: rearrange(x, "b (c n) ... -> b (n c) ... ", c=3),
        #     [log_f, a, b, k, v, q],
        # )

        k_ = k * (-beta)
        a = torch.cat([k_ * torch.exp(log_f), k_], dim=1)
        b = torch.cat([k, k], dim=1)
        k = torch.cat([self.zero, k], dim=1)
        v = torch.cat([self.zero, v], dim=1)
        q = torch.cat([self.zero, q], dim=1)
        log_f = torch.cat([log_f, self.zero], dim=1)
        log_f, a, b, k, v, q = map(
            lambda x: rearrange(x, "b (c n) ... -> b (n c) ... ", c=2),
            [log_f, a, b, k, v, q],
        )

        scale = 1
        if self.causal:
            dtype = q.dtype
            if self.training or use_cache:
                fn = chunk_dplr_delta_rule
            else:
                fn = fused_recurrent_dplr_delta_rule

            output, recurrent_state = fn(
                q=q,
                k=k.to(dtype),
                v=v.to(dtype),
                a=a.to(dtype),
                b=b.to(dtype),
                gk=log_f.to(dtype),
                initial_state=recurrent_state,
                output_final_state=use_cache,
                scale=scale,
                head_first=False,
            )
        else:
            assert False

        output = rearrange(output, "b (n c) ... -> b (c n) ...", c=2)
        output = output[:, -n:]

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
