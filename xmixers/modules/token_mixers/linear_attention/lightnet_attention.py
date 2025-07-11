from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers.cache_utils import Cache
from xopes.ops import lightning_attn_func

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.modules.pes import Lrpe
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params


class LightNetAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_lrpe: bool = True,
        lrpe_type: int = 1,
        base: int = 10000,
        use_output_gate: bool = True,
        token_mixer_norm_type: str = "rmsnorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        scalar_decay: bool = False,
        causal: bool = True,
        gate_act: str = "sigmoid",
        gate_pos: str = "pre",
        use_input_gate: bool = False,
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
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.use_output_gate = use_output_gate

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.scalar_decay = scalar_decay
        if self.scalar_decay:
            self.f_proj = nn.Linear(embed_dim, self.num_heads, bias=bias)

        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
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

        self.use_lrpe = use_lrpe

        if self.use_lrpe:
            self.lrpe = Lrpe(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                lrpe_type=lrpe_type,
                base=base,
            )

        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )

        self.use_input_gate = use_input_gate
        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self.lse_state = torch.empty(0)
        self._init_weights()

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
        **kwargs,
    ):
        b, n, d = x.shape
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # act
        q = self.q_act(q)

        q, k, v = map(
            lambda x: rearrange(
                x, "b n (h d) -> b n h d", d=self.head_dim
            ).contiguous(),
            [q, k, v],
        )

        recurrent_state = None
        q_offset = 0
        if self.scalar_decay:
            shape = (b, 1, self.num_heads)
        else:
            shape = (b, 1, self.num_heads, self.head_dim)

        if self.lse_state.shape[0] == 0 or self.lse_state.shape[0] != b:
            self.lse_state = torch.zeros(shape, device=x.device, dtype=torch.float32)
        lse_state = self.lse_state

        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"][0]
            lse_state = past_key_values[self.layer_idx]["recurrent_state"][1]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        if self.scalar_decay:
            f = self.f_proj(x).to(torch.float32)
        else:
            f = k.to(torch.float32)

        if attention_mask is not None and not attention_mask.all():
            start = q_offset
            if self.scalar_decay:
                attention_mask_ = attention_mask[:, start:].unsqueeze(-1)
            else:
                attention_mask_ = attention_mask[:, start:].unsqueeze(-1).unsqueeze(-1)
            f = f.masked_fill(attention_mask_ == 0, -float("inf"))

        f = torch.cat([lse_state, f], dim=1)
        z = f.float().logcumsumexp(1)
        log_f = (z[:, :-1] - z[:, 1:]).to(k.dtype)
        lse_state = z[:, -1:]

        if self.scalar_decay:
            k = self.k_act(k)
            if self.use_input_gate:
                k = (1 - torch.exp(log_f).unsqueeze(-1)) * k
        else:
            k = torch.exp(k - z[:, 1:])

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k, offset=q_offset)
            if not self.scalar_decay:
                log_f = repeat(log_f, "... d -> ... (g d)", g=2)

        dtype = q.dtype
        q, k, v, log_f = map(lambda x: x.to(dtype), [q, k, v, log_f])

        # left padding
        if attention_mask is not None and not attention_mask.all():
            start = q_offset
            log_f = log_f.masked_fill(attention_mask_ == 0, 0)
            if self.scalar_decay:
                attention_mask_ = attention_mask_.unsqueeze(-1)
            k = k.masked_fill(attention_mask_ == 0, 0)

        if self.causal:
            if self.scalar_decay:
                ld = log_f
                ldk = None
                decay_type = "scalar"
            else:
                ld = None
                ldk = log_f
                decay_type = "vector"

            output, recurrent_state = lightning_attn_func(
                q=q,
                k=k,
                v=v,
                ld=ld,
                ldk=ldk,
                initial_state=recurrent_state,
                decay_type=decay_type,
            )
        else:
            assert False

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=[recurrent_state, lse_state],
                layer_idx=self.layer_idx,
                offset=n,
            )

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        if self.use_output_gate:
            gate = self.output_gate(x)
            output = self.norm(output, gate)
        else:
            output = self.norm(output)

        # out proj
        output = self.o_proj(output)

        return output, past_key_values
