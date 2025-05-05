from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from xopes.ops import lightning_attn_func

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params


class Hgru1(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_output_gate: bool = False,
        token_mixer_norm_type: str = "rmsnorm",
        q_activation: str = "silu",
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
        self.head_dim = head_dim
        self.causal = causal
        self.use_output_gate = use_output_gate
        self.embed_dim = embed_dim

        self.q_proj = nn.Sequential(
            nn.Linear(embed_dim, head_dim, bias=bias),
            nn.Linear(head_dim, 2 * embed_dim, bias=bias),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(embed_dim, head_dim, bias=bias),
            nn.Linear(head_dim, 2 * embed_dim, bias=bias),
        )
        self.v_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        self.o_proj = nn.Linear(2 * embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        num_groups = embed_dim // head_dim
        self.norm = get_norm_fn(token_mixer_norm_type)(
            2 * embed_dim,
            bias=bias,
            gate_act=gate_act,
            gate_pos=gate_pos,
            num_groups=num_groups,
        )

        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, head_dim, bias=bias),
                nn.Linear(head_dim, 2 * embed_dim, bias=bias),
            )

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
        log_f = self.k_proj(x)
        v = self.v_proj(x)

        # act
        q = self.q_act(q)
        f = F.sigmoid(log_f)

        # l + (1 - l) * sigmoid(x)
        if lower_bound is not None:
            f = lower_bound + (1 - lower_bound) * f
            log_f = torch.log(f)
        else:
            log_f = F.logsigmoid(f)
        k = 1 - torch.exp(log_f.float())

        recurrent_state = None
        q_offset = 0
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"][0]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        dtype = q.dtype
        q, k, v, log_f = map(lambda x: x.to(dtype), [q, k, v, log_f])
        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )
        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1)
            v = v.masked_fill(attention_mask_ == 0, 0)
            log_f = log_f.masked_fill(attention_mask_ == 0, 0)

        if self.causal:
            output, recurrent_state = lightning_attn_func(
                q=q,
                k=k,
                v=v,
                ld=log_f,
                initial_state=recurrent_state,
                decay_type="element",
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
        if self.use_output_gate:
            gate = self.output_gate(x)
            output = self.norm(output, gate)
        else:
            output = self.norm(output)

        # out proj
        output = self.o_proj(output)

        return output, past_key_values
