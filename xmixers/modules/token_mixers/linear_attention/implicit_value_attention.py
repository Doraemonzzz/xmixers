import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache
from xopes.ops import kernel_regression_func, lightning_attn_func

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn, l2_norm
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params


class ImplicitValueAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_output_gate: bool = False,
        token_mixer_norm_type: str = "rmsnorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        use_decay: bool = True,
        threshold: float = 0.99,
        use_offset: bool = False,
        causal: bool = True,
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
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.use_output_gate = use_output_gate
        self.threshold = threshold
        self.use_offset = use_offset

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.bet_proj = nn.Linear(embed_dim, num_heads, bias=bias)

        self.use_decay = use_decay
        if self.use_decay:
            self.f_proj = nn.Linear(embed_dim, num_heads, bias=bias)

        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        token_mixer_norm_type = (
            f"{token_mixer_norm_type}_fused_gate"
            if token_mixer_norm_type
            else token_mixer_norm_type
        )
        self.norm = get_norm_fn(token_mixer_norm_type)(
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

        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self._init_weights()

    def extra_repr(self):
        return print_module(self)

    def setup_decay(self):
        if (not self.use_decay) or (not self.use_offset):
            return
        # take x = 0 as median, 1 / (1 + exp(-(median + delta))) = a => 1 + exp(-delta) = 1 / a => exp(-delta) = (1 / a - 1) -> exp(delta) = a / (1 - a) => delta = log(a / (1 - a))
        a = self.threshold
        delta = torch.ones(self.num_heads) * math.log(a / (1 - a))
        if hasattr(self, "delta"):
            if isinstance(self.delta, DTensor):
                self.delta.data.copy_(
                    DTensor.from_local(
                        delta,
                        device_mesh=self.delta.device_mesh,
                    )
                )
            else:
                self.delta.data.copy_(delta)
        else:
            self.delta = nn.Parameter(delta, requires_grad=True)

    def _init_weights(self):
        self.setup_decay()
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        self.setup_decay()
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
        # x: b n d
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.use_decay:
            f = self.f_proj(x)
            if self.use_offset:
                f = f + self.delta
        beta = 2 * F.sigmoid(self.bet_proj(x))

        q, k, v = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim),
            [q, k, v],
        )

        q = self.q_act(q)
        k = self.k_act(k)
        k = l2_norm(k)
        if self.use_decay:
            log_f = F.logsigmoid(f.float())
            decay_type = "scalar"
        else:
            log_f = None
            decay_type = "constant"

        recurrent_state1 = None
        recurrent_state2 = None
        q_offset = 0
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state1 = past_key_values[self.layer_idx]["recurrent_state"][0]
            recurrent_state2 = past_key_values[self.layer_idx]["recurrent_state"][1]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )
        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1).unsqueeze(-1)
            v = v.masked_fill(attention_mask_ == 0, 0)
            if self.use_decay:
                log_f = log_f.masked_fill(attention_mask_.squeeze(-1) == 0, 0)

        if self.causal:
            v, recurrent_state1 = kernel_regression_func(
                q=None,
                k=k,
                v=v,
                ld=log_f,
                beta=beta,
                initial_state=recurrent_state1,
            )

            output, recurrent_state2 = lightning_attn_func(
                q=q,
                k=k * beta.unsqueeze(-1),
                v=v,
                ld=log_f,
                initial_state=recurrent_state2,
                decay_type=decay_type,
            )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=[recurrent_state1, recurrent_state2],
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
