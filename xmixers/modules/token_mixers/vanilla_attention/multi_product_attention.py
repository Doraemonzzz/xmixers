# multi product attention
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params

from ...pes import Lrpe

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None

from xopes.ops.out_product_linear_recurrence import oplr_fn

from xmixers.modules.normalizations import l2_norm


class MultiProductAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        use_lrpe: bool = True,
        layer_idx: int = 0,
        lrpe_type: int = 1,
        base: int = 10000,
        max_position_embeddings: int = 1024,
        token_mixer_init_type: int = 0,
        rescale_type: int = 0,
        num_layers: int = 12,
        mpa_type: int = 0,
        mpa_activation: str = "none",
        head_dim: int = -1,
        gate_type: int = 0,
        init_std: float = 0.02,
        gain: float = 0.02,
        use_l2_norm: bool = False,
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
        self.head_dim = embed_dim // num_heads if head_dim == -1 else head_dim
        mid_dim = self.head_dim * self.num_heads
        self.mpa_type = mpa_type
        self.q_proj = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, 2 * self.head_dim, bias=bias)
        if self.mpa_type == 0:
            self.kv_head_proj = nn.Linear(embed_dim, 2 * num_heads, bias=bias)
        else:
            self.kv_head = nn.Parameter(
                torch.randn(2 * num_heads) * 0.1, requires_grad=True
            )
        self.gate_type = gate_type

        self.act = get_activation_fn(mpa_activation)

        self.out_proj = nn.Linear(mid_dim, embed_dim, bias=bias)

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                lrpe_type=lrpe_type,
                base=base,
                max_position_embeddings=max_position_embeddings,
            )

        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self.use_l2_norm = use_l2_norm
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
        **kwargs,
    ):
        b, n, d = x.shape
        # x: b n d
        # linear map
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        if self.mpa_type == 0:
            kv_head = self.act(self.kv_head_proj(x))
        else:
            kv_head = self.act(self.kv_head)

        k, v = kv.chunk(2, dim=-1)
        k_head, v_head = kv_head.chunk(2, dim=-1)

        # TODO: add a kernel here
        if self.gate_type == 0:
            k, v = map(
                lambda arr: torch.einsum("... d, ... h -> ... h d", arr[0], arr[1]),
                [(k, k_head), (v, v_head)],
            )
        elif self.gate_type == 1:
            index = 1 + torch.arange(
                0, n, dtype=torch.float32, device=k.device
            ).unsqueeze(-1).unsqueeze(-1)
            # cumsum
            k = (oplr_fn(k_head, k, log_decay=None, decay_type="nd")).to(q.dtype)
            v = (oplr_fn(v_head, v, log_decay=None, decay_type="nd")).to(q.dtype)
            if self.use_l2_norm:
                k = l2_norm(k)
        elif self.gate_type == 2:
            k = oplr_fn(k_head, k, log_decay=None, decay_type="ddd")
            v = oplr_fn(v_head, v, log_decay=None, decay_type="ddd")
            if self.use_l2_norm:
                k = l2_norm(k)
        elif self.gate_type == 3:
            k = oplr_fn(k_head, k, log_decay=None, decay_type="ddd")
            v = torch.einsum("... d, ... h -> ... h d", v, v_head)
        elif self.gate_type == 4:
            k = torch.einsum("... d, ... h -> ... h d", k, k_head)
            v = oplr_fn(v_head, v, log_decay=None, decay_type="ddd")
        elif self.gate_type == 5:
            index = 1 + torch.arange(
                0, n, dtype=torch.float32, device=k.device
            ).unsqueeze(-1).unsqueeze(-1)
            k = (oplr_fn(k_head, k, log_decay=None, decay_type="ddd") / index).to(
                q.dtype
            )
            v = oplr_fn(v_head, v, log_decay=None, decay_type="ddd")

        # for lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        # TODO: upate cache update
        if past_key_values is not None:
            k, v, k_head, v_head = past_key_values.update(
                mpa_state=(k, v, k_head, v_head),
                layer_idx=self.layer_idx,
                offset=x.shape[-2],
            )["mpa_state"]

        # construct k, v cache
        # todo: add a fusion here
        q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads)

        q, k, v = map(
            lambda x: rearrange(x, "... n h d -> ... h n d"),
            [q, k, v],
        )

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k)

        # if (
        #     attention_mask is None or attention_mask.all()
        # ):  # if attention mask is None or all elements are True, use sdpa
        if True:
            # use causal when training or evaluation(not for generation) or prefill
            is_causal = True if self.training or q.shape[-2] == k.shape[-2] else False
            # output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            q, k, v = map(lambda x: rearrange(x, "... h n d -> ... n h d"), [q, k, v])
            dtype = q.dtype
            if dtype == torch.float32:
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)
            output = flash_attn_func(q, k, v, causal=is_causal).to(dtype)
            output = rearrange(output, "... n h d -> ... h n d")
        else:
            assert False, "flash_attn_varlen_qkvpacked_func current not support"

        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")
        # outproj
        output = self.out_proj(output)

        return output, past_key_values
