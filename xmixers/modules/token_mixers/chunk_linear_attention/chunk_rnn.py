import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache
from xopes.ops import chunk_rnn_parallel_fn, chunk_rnn_sequential_fn

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params

from ...pes import Lrpe


class ChunkRnn(nn.Module):
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
        causal: bool = True,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.01,
        gate_act: str = "sigmoid",
        gate_pos: str = "pre",
        threshold: float = 0.99,
        # chunk params
        chunk_type: int = 0,
        gradient_type: int = 0,
        use_initial_state: bool = False,
        use_scale: bool = False,
        chunk_size: int = 128,
        # lrpe
        use_lrpe: bool = True,
        lrpe_type: int = 6,
        base: int = 10000,
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
        self.use_initial_state = use_initial_state
        self.use_scale = use_scale
        self.chunk_size = chunk_size
        self.chunk_type = chunk_type
        self.gradient_type = gradient_type
        self.decay_dim = num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.f_proj = nn.Linear(embed_dim, self.num_heads, bias=bias)

        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.threshold = threshold
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

        if self.use_initial_state:
            self.state = nn.Parameter(
                torch.zeros(self.num_heads, self.head_dim, self.head_dim),
                requires_grad=True,
            )
        else:
            self.state = None

        if self.use_scale:
            self.scale = nn.Parameter(
                torch.ones(self.num_heads, expand_ratio), requires_grad=True
            )
        else:
            self.scale = None

        self.gradient_type = gradient_type
        if self.gradient_type == 0:
            self.gradient_fn = chunk_rnn_parallel_fn
        else:
            self.gradient_fn = chunk_rnn_sequential_fn

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                lrpe_type=lrpe_type,
                base=base,
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
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        b, n, d = x.shape
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        log_f = F.logsigmoid(self.f_proj(x) + self.delta)

        # act
        q = self.q_act(q)
        k = self.k_act(k)

        q, k, v = map(
            lambda x: rearrange(x, "b n (h d) -> b n h d", d=self.head_dim),
            [q, k, v],
        )

        # lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k, offset=q_offset)

        if self.state is not None:
            recurrent_state = repeat(self.state, "h d e -> b h d e", b=x.shape[0])
        else:
            recurrent_state = None

        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"]

        dtype = q.dtype
        q, k, v = map(lambda x: x.to(dtype), [q, k, v])
        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )
        # TODO: softmax attn mask has not been treated
        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1).unsqueeze(-1)
            v = v.masked_fill(attention_mask_ == 0, 0)
            log_f = log_f.masked_fill(attention_mask_ == 0, 0)

        if self.causal:
            output, recurrent_state = self.gradient_fn(
                q=q,
                k=k,
                v=v,
                log_f=log_f,
                initial_state=recurrent_state,
                chunk_size=self.chunk_size,
            )
        else:
            assert False

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=[recurrent_state],
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
