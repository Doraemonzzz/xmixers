from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache

from xmixers.modules.pes import Lrpe
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except:
    flash_attn_func = None

import math

import torch.nn.functional as F
from xopes.ops import kernel_regression_func

from xmixers.modules.normalizations import l2_norm
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params

from .utils import _pad_input, _upad_input


class KernelRegressionAttention(nn.Module):
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
        max_position_embeddings: int = 1024,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        window_size: int = -1,
        use_decay: bool = False,
        use_kernel_regression: bool = True,
        scale_type: int = 0,
        threshold: float = 0.99,
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

        self.layer_idx = layer_idx
        self.kv_heads = kv_heads
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        if self.kv_heads == -1:
            kv_dim = embed_dim
        else:
            kv_dim = self.kv_heads * self.head_dim
        self.threshold = threshold

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.use_kernel_regression = use_kernel_regression
        self.use_decay = use_decay

        if self.use_kernel_regression:
            self.bet_proj = nn.Linear(embed_dim, num_heads, bias=bias)
            if self.use_decay:
                self.f_proj = nn.Linear(embed_dim, num_heads, bias=bias)

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
        self.scale_type = scale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self._init_weights()

    def extra_repr(self):
        return print_module(self)

    def setup_decay(self):
        if not self.use_decay:
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
        **kwargs,
    ):
        b, n, d = x.shape
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.use_decay:
            f = self.f_proj(x) + self.delta

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v],
        )
        k = l2_norm(k)
        if self.use_kernel_regression:
            beta = 2 * F.sigmoid(self.bet_proj(x))
            if self.use_decay:
                log_f = F.logsigmoid(f.float())
            else:
                log_f = None

        if self.scale_type == 0:
            scale = q.shape[-1] ** -0.5
        elif self.scale_type == 1:
            scale = 1

        # lrpe
        recurrent_state = None
        q_offset = 0
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            if self.use_kernel_regression:
                recurrent_state = past_key_values[self.layer_idx]["recurrent_state"][0]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k, offset=q_offset)

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

        if self.use_kernel_regression:
            v, recurrent_state = kernel_regression_func(
                q=None,
                k=k,
                v=v,
                ld=log_f,
                beta=beta,
                initial_state=recurrent_state,
            )

        if self.use_kernel_regression:
            k = k * beta.unsqueeze(-1)

        # cache update
        if past_key_values is not None:
            k, v = past_key_values.update(
                recurrent_state=[recurrent_state],
                attn_state=(k, v),
                layer_idx=self.layer_idx,
                offset=n,
            )["attn_state"]

        causal = True if self.training or q.shape[-3] == k.shape[-3] else False
        window_size = (self.window_size, 0) if self.window_size > 0 else (-1, -1)

        # only use cu_seqlens in training
        cu_seqlens = kwargs.get("cu_seqlens", None)
        if (cu_seqlens is not None) or (
            attention_mask is not None and not attention_mask.all()
        ):
            if cu_seqlens is not None:  # flame training stage
                cu_seqlens_q = cu_seqlens_k = cu_seqlens
                max_seqlen_q = max_seqlen_k = n
                q = q.squeeze(0)
                k = k.squeeze(0)
                v = v.squeeze(0)
            else:
                q, k, v, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
                    q=q, k=k, v=v, attention_mask=attention_mask, q_len=n
                )
                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_q, max_seqlen_k = max_seq_lens

            output = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=causal,
                window_size=window_size,
                softmax_scale=scale,
            )

            if cu_seqlens is None:
                output = _pad_input(output, indices_q, b, n)
            else:
                output = output.unsqueeze(0)
        else:
            output = flash_attn_func(
                q, k, v, causal=causal, window_size=window_size, softmax_scale=scale
            )

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        # outproj
        output = self.o_proj(output)

        return output, past_key_values
