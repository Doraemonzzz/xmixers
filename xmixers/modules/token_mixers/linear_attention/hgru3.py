# Hgru3: Hgru2 with negative decay
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.gla import chunk_gla, fused_recurrent_gla
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params


class Hgru3(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expand_ratio: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        token_mixer_norm_type: str = "rmsnorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        threshold: float = 0.99,
        causal: bool = True,
        use_dense_memory: bool = False,
        scalar_decay: bool = False,
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
        self.expand_ratio = expand_ratio
        self.causal = causal
        self.use_dense_memory = use_dense_memory
        self.use_output_gate = use_output_gate

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.scalar_decay = scalar_decay
        self.decay_dim = embed_dim
        if self.scalar_decay:
            self.decay_dim = embed_dim // expand_ratio
            self.f_proj = nn.Linear(embed_dim, self.decay_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.threshold = threshold

        num_groups = embed_dim // expand_ratio
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
            num_groups=num_groups,
        )

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
        self.setup_decay()
        self.apply(self._initialize_weights)

    def extra_repr(self):
        return print_module(self)

    def setup_decay(self):
        # take x = 0 as median, 1 / (1 + exp(-(median + delta))) = a => 1 + exp(-delta) = 1 / a => exp(-delta) = (1 / a - 1) -> exp(delta) = a / (1 - a) => delta = log(a / (1 - a))
        a = self.threshold
        delta = torch.ones(self.decay_dim) * math.log(a / (1 - a))
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
        # x: b n d
        # linear map
        q = self.q_proj(x)
        v = self.v_proj(x)
        if self.scalar_decay:
            log_f = F.logsigmoid(self.f_proj(x) + self.delta)
            k = self.k_proj(x)
            k = self.k_act(k)
        else:
            log_f = F.logsigmoid(self.k_proj(x) + self.delta)
            k = 1 - torch.exp(log_f)

        # act
        q = self.q_act(q)

        # h is num_head, d is head dimension
        q, k, v = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [q, k, v],
        )
        if not self.scalar_decay:
            log_f = rearrange(log_f, "... (h d) -> ... h d", d=self.expand_ratio)

        recurrent_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"]

        scale = 1
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
                scale=scale,
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
