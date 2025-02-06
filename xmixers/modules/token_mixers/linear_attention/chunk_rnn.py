from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache
from xopes.ops import chunk_rnn_parallel_fn, chunk_rnn_sequential_fn

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn, l2_norm
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params


class ChunkRnn(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expand_ratio: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_output_gate: bool = False,
        token_mixer_norm_type: str = "layernorm",
        q_activation: str = "silu",
        causal: bool = True,
        token_mixer_init_type: int = 0,
        rescale_type: int = 0,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.02,
        # chunk params
        chunk_type: int = 0,
        gradient_type: int = 0,
        use_init_weights: bool = False,
        use_scale: bool = False,
        chunk_size: int = 128,
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
        self.num_heads = embed_dim // expand_ratio
        self.use_output_gate = use_output_gate
        self.use_init_weights = use_init_weights
        self.use_scale = use_scale
        self.chunk_size = chunk_size
        self.chunk_type = chunk_type
        self.gradient_type = gradient_type

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        num_groups = embed_dim // expand_ratio
        self.norm = get_norm_fn(token_mixer_norm_type)(
            embed_dim, bias=False, num_groups=num_groups
        )

        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, expand_ratio, bias=bias),
                nn.Linear(expand_ratio, embed_dim, bias=bias),
            )

        if self.use_init_weights:
            self.state = nn.Parameter(
                torch.zeros(self.num_heads, expand_ratio, expand_ratio),
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

        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self.apply(self._initialize_weights)

    def extra_repr(self):
        return print_module(self)

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
        # x: b n d
        # linear map
        # !!! not q, log_f, v, this is for align with internel version
        q, log_f, v = self.in_proj(x).chunk(3, dim=-1)

        # act
        q = self.q_act(q)
        f = F.sigmoid(log_f)

        # todo: make a fusion here
        # l + (1 - l) * sigmoid(x)
        if lower_bound is not None:
            f = lower_bound + (1 - lower_bound) * f
            log_f = torch.log(f)
        else:
            log_f = F.logsigmoid(f)

        k = 1 - f
        if self.gradient_type == 1:
            k = l2_norm(k)

        q, k, v, log_f = map(
            lambda x: rearrange(
                x, "b n (h d) -> b n h d", d=self.expand_ratio
            ).contiguous(),
            [q, k, v, log_f],
        )

        if self.state is not None:
            recurrent_state = repeat(self.state, "h d e -> b h d e", b=x.shape[0])
        else:
            recurrent_state = None

        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"]

        dtype = q.dtype
        q, k, v, log_f = map(lambda x: x.to(dtype), [q, k, v, log_f])
        if self.causal:
            output, recurrent_state = self.gradient_fn(
                q=q,
                k=k,
                v=v,
                log_f=log_f,
                initial_state=recurrent_state,
                scale=self.scale,
                gradient_type=self.gradient_type,
                chunk_size=self.chunk_size,
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

        output = self.norm(output)

        # out proj
        output = self.out_proj(output)

        return output, past_key_values
