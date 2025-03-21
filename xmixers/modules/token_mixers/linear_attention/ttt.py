from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.ttt import chunk_ttt_linear
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn, l2_norm
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params


class TTT(nn.Module):
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
        norm_k: bool = True,
        beta_activation: str = "neg",
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

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # !!! dont use beta as name in hf: https://github.com/huggingface/transformers/issues/29554
        self.bet_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        self.ln_weight = nn.Parameter(
            torch.ones(num_heads, self.head_dim), requires_grad=True
        )
        self.ln_bias = nn.Parameter(
            torch.zeros(num_heads, self.head_dim), requires_grad=True
        )
        self.beta_activation = beta_activation
        self.norm_k = norm_k

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
        **kwargs,
    ):
        b, n, d = x.shape
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v],
        )

        beta = F.sigmoid(self.bet_proj(x))
        if self.beta_activation == "neg":
            beta = beta * 2

        # act
        q = self.q_act(q)
        if self.norm_k:
            k = l2_norm(k)
        else:
            k = self.k_act(k)

        recurrent_state = None
        q_offset = 0
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"][0]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )
        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1).unsqueeze(-1)
            k = k.masked_fill(attention_mask_ == 0, 0)

        scale = 1
        if self.causal:
            if self.training or recurrent_state is None:  # training or prefilling
                fn = chunk_ttt_linear
            else:
                fn = chunk_ttt_linear

            output, recurrent_state, recurrent_state_ = fn(
                q=q,
                k=k,
                v=v,
                w=self.ln_weight,
                b=self.ln_bias,
                eta=beta,
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
