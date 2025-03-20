from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params


class GatedSlotAttention(nn.Module):
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
        num_slots: int = 64,
        causal: bool = True,
        gate_act: str = "sigmoid",
        gate_pos: str = "pre",
        max_position_embeddings: int = 1024,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.01,
        use_initial_state: bool = False,
        **kwargs,
    ):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_slots = num_slots

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.f_proj = nn.Sequential(
            nn.Linear(embed_dim, self.head_dim, bias=bias),
            nn.Linear(self.head_dim, self.num_slots * self.num_heads, bias=bias),
        )
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
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
        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.causal = causal

        self.use_output_gate = use_output_gate
        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )

        self.use_initial_state = use_initial_state
        if self.use_initial_state:
            self.initial_state = nn.Parameter(
                torch.zeros(self.num_heads, self.head_dim, self.head_dim)
            )
        else:
            self.initial_state = None

        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self._init_weights()

    def _init_weights(self):
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        return _initialize_weights(self, module)

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
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        f = self.f_proj(x)

        # l + (1 - l) * sigmoid(x)
        if lower_bound is not None:
            f = lower_bound + (1 - lower_bound) * F.sigmoid(f)
            log_f = torch.log(f)
        else:
            log_f = F.logsigmoid(f)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v],
        )
        log_f = rearrange(log_f, "... n (h d) -> ... n h d", d=self.num_slots)

        # act
        q = self.q_act(q)
        k = self.k_act(k)
        s = 1 - torch.exp(log_f)

        recurrent_state = None
        q_offset = 0
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"][0]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        dtype = q.dtype
        q, k, v, log_f, s = map(lambda x: x.to(dtype), [q, k, v, log_f, s])
        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )
        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1).unsqueeze(-1)
            k = k.masked_fill(attention_mask_ == 0, 0)
            log_f = log_f.masked_fill(attention_mask_ == 0, 0)
            s = s.masked_fill(attention_mask_ == 0, 0)

        scale = 1

        if self.causal:
            if self.training or recurrent_state is None:  # training or prefilling
                fn = chunk_gsa
            else:
                fn = fused_recurrent_gsa
            output, recurrent_state = fn(
                q=q,
                k=k,
                v=v,
                s=s,
                g=log_f,
                scale=scale,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                head_first=False,
            )
        else:
            assert False, "not implemented"

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=[recurrent_state],
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

        # outproj
        output = self.o_proj(output)

        return output, past_key_values
