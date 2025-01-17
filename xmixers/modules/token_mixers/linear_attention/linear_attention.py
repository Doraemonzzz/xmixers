from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params

from ...pes import Lrpe


class LinearAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_heads: int = -1,
        bias: bool = False,
        use_lrpe: bool = True,
        layer_idx: int = 0,
        lrpe_type: int = 1,
        base: int = 10000,
        use_output_gate: bool = True,
        norm_type: str = "layernorm",
        linear_activation: str = "silu",
        causal: bool = True,
        use_dense_memory: bool = False,
        max_position_embeddings: int = 1024,
        token_mixer_init_type: int = 0,
        rescale_type: int = 0,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.layer_idx = layer_idx
        self.kv_heads = kv_heads
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.kv_heads == -1:
            kv_dim = embed_dim
        else:
            kv_dim = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.norm = get_norm_fn(norm_type)(embed_dim)
        self.act = get_activation_fn(linear_activation)
        self.causal = causal

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                lrpe_type=lrpe_type,
                base=base,
                act=linear_activation,
            )

        self.use_output_gate = use_output_gate
        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )

        self.use_dense_memory = use_dense_memory
        if self.use_dense_memory:
            self.alpha_proj = nn.Linear(embed_dim, 1, bias=bias)
            self.beta_proj = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )

        self.causal_mask = None
        self.max_position_embeddings = max_position_embeddings
        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        return _initialize_weights(self, module)

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        # x: b n d
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v],
        )

        # lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        # cache update
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k)

        if self.causal:
            if self.causal_mask is None:
                self.causal_mask = (
                    torch.tril(
                        torch.ones(
                            self.max_position_embeddings, self.max_position_embeddings
                        )
                    )
                ).to(q)
            energy = torch.einsum("... n h d, ... m h d -> ... h n m", q, k)
            # use causal when training or evaluation(not for generation) or prefill
            is_causal = True if self.training or (q.shape[1] == k.shape[1]) else False
            if is_causal:
                n = k.shape[1]
                causal_mask = self.causal_mask[:n, :n]
                # print(q.shape, energy.shape, v.shape, causal_mask.shape)
                energy = energy * causal_mask
            output = torch.einsum("... h n m, ... m h d -> ... n h d", energy, v)
        else:
            kv = torch.einsum("... n h d, ... n h e -> ... h d e", k, v)
            output = torch.einsum("... n h d, ... h d e -> ... n h e", q, kv)

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")

        if self.use_output_gate:
            output_gate = F.sigmoid(self.output_gate(x))
            output = output * output_gate

        # use norm here for better parallel when using tp
        output = self.norm(output)

        # outproj
        output = self.out_proj(output)

        return output, past_key_values
