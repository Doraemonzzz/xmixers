# flex attention: https://pytorch.org/blog/flexattention/
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.forgetting_attn.parallel import parallel_forgetting_attn
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache

from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params

from .utils import _pad_input, _unpad_input


class ForgettingAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_heads: int = -1,
        bias: bool = False,
        layer_idx: int = 0,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        window_size: int = -1,
        init_std: float = 0.02,
        gain: float = 0.01,
        threshold: float = 0.99,
        use_offset: bool = False,
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
        self.use_offset = use_offset

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.f_proj = nn.Linear(embed_dim, num_heads, bias=bias)

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
        if not self.use_offset:
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
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.use_offset:
            f = self.f_proj(x) + self.delta
        else:
            f = self.f_proj(x)
        log_f = F.logsigmoid(f)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v],
        )

        if past_key_values is not None:
            past_key_values.get_seq_length(self.layer_idx)

        # cache update
        if past_key_values is not None:
            k, v, log_f = past_key_values.update(
                attn_state=(k, v, log_f),
                layer_idx=self.layer_idx,
                offset=n,
            )["attn_state"]

        cu_seqlens = None

        if attention_mask is not None and not attention_mask.all():
            q, (k, v, log_f), indices_q, cu_seqlens, max_seq_lens = _unpad_input(
                q=q,
                states=(k, v, log_f),
                attention_mask=attention_mask,
                q_len=n,
            )
            _, cu_seqlens_k = cu_seqlens
            cu_seqlens = cu_seqlens_k
            max_seqlen_q, max_seqlen_k = max_seq_lens
            if max_seqlen_q != max_seqlen_k:
                assert max_seqlen_q == 1, "only support q_len == 1 for decoding"
                output = attn_decoding_one_step(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    log_f.unsqueeze(0),
                    cu_seqlens=cu_seqlens,
                )
            else:
                output = parallel_forgetting_attn(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    log_f.unsqueeze(0),
                    cu_seqlens=cu_seqlens,
                )

            output = _pad_input(output.squeeze(0), indices_q, b, n)
        else:
            if n == 1:
                attention_mask = torch.ones(b, k.shape[1], dtype=torch.int32).to(
                    q.device
                )

                q, (k, v, log_f), indices_q, cu_seqlens, max_seq_lens = _unpad_input(
                    q=q,
                    states=(k, v, log_f),
                    attention_mask=attention_mask,
                    q_len=n,
                )
                _, cu_seqlens_k = cu_seqlens
                cu_seqlens = cu_seqlens_k
                max_seqlen_q, max_seqlen_k = max_seq_lens
                output = attn_decoding_one_step(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    log_f.unsqueeze(0),
                    cu_seqlens=cu_seqlens,
                )
                output = _pad_input(output.squeeze(0), indices_q, b, n)
            else:
                output = parallel_forgetting_attn(q, k, v, log_f, cu_seqlens=cu_seqlens)

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        # outproj
        output = self.o_proj(output)

        return output, past_key_values
