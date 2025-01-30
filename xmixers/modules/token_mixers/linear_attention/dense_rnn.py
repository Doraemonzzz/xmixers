from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from fla.ops.generalized_delta_rule import (
    chunk_dplr_delta_rule,
    fused_recurrent_dplr_delta_rule,
)
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params


class DenseRnn(nn.Module):
    """
    S[t] = (I - k * k ^ T) * Diag(f) * (I - k * k ^ T) * S[t-1] + k * v ^ T
    this is equivalent to the following:
    S[t, 1] = (I - k * k ^ T) * S[t-1] + 0 * 0 ^ T
    S[t, 2] = (Diag(f) - 0 * 0 ^ T) * S[t, 1] + 0 * 0 ^ T
    S[t, 3] = (I - k * k ^ T) * S[t, 2] + k * v ^ T
    Since dplr has the following form:
        S[t] = (D + a b ^ T) * S[t-1] + k * v ^ T
    We can use the following form to implement this:
        D = concat([0, f, 0])
        a = concat([k, 0, k])
        b = a
        k = concat([0, 0, k])
        v = concat([0, 0, v])
        q = concat([0, 0, q])
    to implement this.
    """

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
        scaler_decay: bool = False,
        qkv_norm_type: int = 2,
        norm_q: bool = False,
        norm_v: bool = False,
        causal: bool = True,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.01,
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
        self.scaler_decay = scaler_decay
        self.head_dim = embed_dim // num_heads
        if self.scaler_decay:
            self.f_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        else:
            self.f_proj = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )
        self.use_gamma = use_gamma
        if self.use_gamma:
            self.gamma_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.v_act = get_activation_fn(v_activation)
        self.gamma_activation = gamma_activation
        self.norm = get_norm_fn(norm_type)(embed_dim, bias=bias)

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
        f = self.f_proj(x)

        if self.zero.shape[0] == 0 or self.zero.shape != torch.Size([b, n, h, d // h]):
            self.zero = torch.zeros(b, n, h, d // h).to(q)
        # l + (1 - l) * sigmoid(x)
        if lower_bound is not None:
            f = lower_bound + (1 - lower_bound) * F.sigmoid(f)
            log_f = torch.log(f)
        else:
            log_f = F.logsigmoid(f)
        # b n h 1
        if self.use_gamma:
            gamma = F.sigmoid(self.gamma_proj(x)).unsqueeze(-1)
            if self.gamma_activation == "neg":
                gamma *= 2
        else:
            gamma = 2
        gamma = -gamma
        # act
        q = self.q_act(q)
        k = self.k_act(k)
        v = self.v_act(v)

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

        if not self.scaler_decay:
            log_f = rearrange(log_f, "... (h d) -> ... h d", d=self.head_dim)
        else:
            log_f = repeat(log_f, "... h -> ... h d", d=self.head_dim)

        # D = concat([0, f, 0])
        # a = concat([k, 0, k])
        # b = a
        # k = concat([0, 0, k])
        # v = concat([0, 0, v])
        # q = concat([0, 0, q])
        k_ = k * gamma
        log_f = torch.cat([self.zero, log_f, self.zero], dim=1)
        a = torch.cat([k_, self.zero, k_], dim=1)
        b = torch.cat([k, self.zero, k], dim=1)
        k = torch.cat([self.zero, self.zero, k], dim=1)
        v = torch.cat([self.zero, self.zero, v], dim=1)
        q = torch.cat([self.zero, self.zero, q], dim=1)

        log_f, a, b, k, v, q = map(
            lambda x: rearrange(x, "b (c n) ... -> b (n c) ... ", c=3),
            [log_f, a, b, k, v, q],
        )

        # TODO: update this
        recurrent_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"]

        if self.causal:
            # Unitary update
            if self.training or use_cache:
                fn = chunk_dplr_delta_rule
            else:
                fn = fused_recurrent_dplr_delta_rule

            dtype = q.dtype

            output, recurrent_state = fn(
                q=q,
                k=k.to(dtype),
                v=v.to(dtype),
                a=a.to(dtype),
                b=b.to(dtype),
                gk=log_f.to(dtype),
                initial_state=recurrent_state,
                output_final_state=use_cache,
                scale=1,
                head_first=False,
            )
        else:
            assert False

        output = output[:, -n:]

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
