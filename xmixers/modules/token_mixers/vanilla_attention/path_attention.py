import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.path_attn.parallel import parallel_path_attention
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache
from xopes.ops import cumsum_fn

from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params

from .utils import _pad_input, _unpad_input

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None

import math

import torch.nn.functional as F

from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params

from .utils import _pad_input, _unpad_input


class PathAttention(nn.Module):
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
        threshold: float = 0.99,
        max_position_embeddings: int = 1024,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        window_size: int = -1,
        init_std: float = 0.02,
        gain: float = 0.01,
        use_decay: bool = False,
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
        self.use_decay = use_decay

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.w_proj = nn.Sequential(
            nn.Linear(embed_dim, self.head_dim, bias=bias),
            nn.Linear(self.head_dim, kv_dim, bias=bias),
        )
        self.bet_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        if self.use_decay:
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
        w = self.w_proj(x)
        beta = 2 * F.sigmoid(self.bet_proj(x))
        log_f = None
        if self.use_decay:
            f = self.f_proj(x) + self.delta
            log_f = F.logsigmoid(f.float())

        q, k, v, w = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v, w],
        )
        from fla.modules.l2norm import l2_norm

        w = l2_norm(w)

        if past_key_values is not None:
            past_key_values.get_seq_length(self.layer_idx)

        causal = True if self.training or q.shape[-3] == k.shape[-3] else False
        (self.window_size, 0) if self.window_size > 0 else (-1, -1)
        training = (attention_mask is None) and (
            past_key_values is None
        )  # todo: update this

        if training:  # Training
            output, _ = parallel_path_attention(q=q, k=k, v=v, w=w, beta=beta, g=log_f)

        else:  # Prefilling or decoding
            assert (
                self.training is False
            ), "attention mask is not supported in training. Please use variable length input."
            try:
                last_state = past_key_values[self.layer_idx]
            except KeyError:
                last_state = None

            if last_state is not None:  # Decoding
                if self.use_decay:
                    past_k, past_v, past_log_f = last_state["attn_state"]
                else:
                    past_k, past_v = last_state["attn_state"]

                def rank_one_update(k, w, beta):
                    original_dtype = k.dtype
                    k = k.float()
                    w = w.float()
                    beta = beta.float()
                    k = k - beta[..., None].float() * (k * w).sum(-1, keepdim=True) * w
                    return k.to(original_dtype)

                past_k = rank_one_update(past_k, w, beta)
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
                log_f = (
                    torch.cat([past_log_f, log_f], dim=1) if self.use_decay else None
                )

                past_key_values[self.layer_idx]["attn_state"] = (
                    (k, v, log_f) if log_f is not None else (k, v)
                )
                past_key_values.update(layer_idx=self.layer_idx, offset=n)

                if attention_mask is not None:
                    if self.use_decay:
                        (
                            q,
                            (k, v, log_f),
                            indices_q,
                            cu_seqlens,
                            max_seq_lens,
                        ) = _unpad_input(
                            q=q,
                            states=(k, v, log_f),
                            attention_mask=attention_mask,
                            q_len=n,
                        )
                        max_seqlen_q, max_seqlen_k = max_seq_lens
                    else:
                        q, (k, v), indices_q, cu_seqlens, max_seq_lens = _unpad_input(
                            q=q, states=(k, v), attention_mask=attention_mask, q_len=n
                        )
                        max_seqlen_q, max_seqlen_k = max_seq_lens
                    _, cu_seqlens = cu_seqlens
                    assert max_seqlen_q == 1, "only support q_len == 1 for decoding"
                    output = attn_decoding_one_step(
                        q.unsqueeze(0),
                        k.unsqueeze(0),
                        v.unsqueeze(0),
                        log_f.unsqueeze(0) if log_f is not None else None,
                        cu_seqlens=cu_seqlens,
                    )  # reduced to fox's decoding
                    output = _pad_input(output.squeeze(0), indices_q, b, n)
                else:
                    attention_mask_ = torch.ones(b, k.shape[1], dtype=torch.int32).to(
                        q.device
                    )
                    seqlens = attention_mask_.sum(-1, dtype=torch.int32)
                    cu_seqlens = F.pad(cumsum_fn(seqlens, dim=0), (1, 0))
                    output = attn_decoding_one_step(
                        q, k, v, log_f, cu_seqlens=cu_seqlens
                    )
            else:  # Prefilling
                if attention_mask is None:
                    attention_mask = torch.ones(b, k.shape[1], dtype=torch.int32).to(
                        q.device
                    )

                if n == 1:
                    seqlens = attention_mask.sum(-1, dtype=torch.int32)
                    cu_seqlens = F.pad(cumsum_fn(seqlens, dim=0), (1, 0))
                    output = attn_decoding_one_step(
                        q, k, v, log_f, cu_seqlens=cu_seqlens
                    )

                    past_key_values.update(
                        attn_state=(k, v, log_f) if log_f is not None else (k, v),
                        layer_idx=self.layer_idx,
                        offset=n,
                    )
                else:
                    v_cache = v.clone()
                    log_f_cache = log_f.clone() if self.use_decay else None
                    if self.use_decay:
                        (
                            q,
                            (k, v, w, beta, log_f),
                            indices_q,
                            cu_seqlens,
                            max_seq_lens,
                        ) = _unpad_input(
                            q=q,
                            states=(k, v, w, beta, log_f),
                            attention_mask=attention_mask,
                            q_len=n,
                        )
                    else:
                        (
                            q,
                            (k, v, w, beta),
                            indices_q,
                            cu_seqlens,
                            max_seq_lens,
                        ) = _unpad_input(
                            q=q,
                            states=(k, v, w, beta),
                            attention_mask=attention_mask,
                            q_len=n,
                        )
                    max_seqlen_q, max_seqlen_k = max_seq_lens
                    assert (
                        max_seqlen_q == max_seqlen_k
                    ), "max_seqlen_q should be equal to max_seqlen_k in prefilling"
                    _, cu_seqlens = cu_seqlens
                    output, k_cache = parallel_path_attention(
                        q=q.unsqueeze(0),
                        k=k.unsqueeze(0),
                        v=v.unsqueeze(0),
                        w=w.unsqueeze(0),
                        beta=beta.unsqueeze(0),
                        g=log_f.unsqueeze(0) if log_f is not None else None,
                        cu_seqlens=cu_seqlens,
                        use_cache=True,
                    )

                    k_cache = _pad_input(k_cache.squeeze(0), indices_q, b, n)
                    past_key_values.update(
                        attn_state=(k_cache, v_cache, log_f_cache)
                        if log_f_cache is not None
                        else (k_cache, v_cache),
                        layer_idx=self.layer_idx,
                        offset=n,
                    )
                    output = _pad_input(output.squeeze(0), indices_q, b, n)

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        # outproj
        output = self.o_proj(output)

        return output, past_key_values
