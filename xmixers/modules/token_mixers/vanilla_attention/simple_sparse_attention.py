from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache

from xmixers.modules.pes import Lrpe
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_params

try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except:
    flash_attn_func = None
    index_first_axis = None
    pad_input = None
    unpad_input = None


class SimpleSparseAttention(nn.Module):
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
        chunk_size: int = 128,
        token_mixer_top_k: int = 2,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        window_size: int = -1,
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
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.g_proj = nn.Linear(embed_dim, 2 * self.num_heads, bias=bias)
        self.chunk_size = chunk_size
        self.top_k = token_mixer_top_k

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
        self.mask = torch.empty(0)
        self._init_weights()

    def _init_weights(self):
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        return _initialize_weights(self, module)

    def _get_mask(self, n, c, device):
        g = (n + c - 1) // c
        index = torch.arange(n, dtype=torch.int32, device=device).unsqueeze(1)
        array = torch.arange(g, dtype=torch.int32, device=device).unsqueeze(0)
        left_thresh = c * (array)
        right_thresh = c * (1 + array)
        return (index >= left_thresh) & (index < right_thresh)

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        b, n, d = x.shape
        c = self.chunk_size
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... n h d", d=self.head_dim),
            [q, k, v],
        )
        g = rearrange(g, "... (h g) -> ... h g", g=2)
        g = F.softmax(g, dim=-1)

        # lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k, offset=q_offset)

        # cache update
        if past_key_values is not None:
            k, v = past_key_values.update(
                attn_state=(k, v),
                layer_idx=self.layer_idx,
                offset=n,
            )["attn_state"]

        # Step 1, compute compressed k and v
        # use hgrn1 to do this
        q_, k_, v_ = map(
            lambda x: rearrange(x, "b (g c) h d -> b g c h d", c=c),
            [q, k, v],
        )
        g = (n + c - 1) // c
        k_compress = k_.mean(dim=-3)

        # Step 2, use compressed k to select chunk
        score = torch.einsum("b n h d, b g h d -> b h n g", q, k_compress)
        # mask current chunk to avoid data leakage
        if self.mask.shape[0] == 0:
            self.mask = self._get_mask(n, c, x.device)
        score = score.masked_fill(self.mask, -float("inf"))
        indices = torch.topk(score, self.top_k, dim=-1)[1]
        # TODO: update this
        k_select = torch.gather(k_, dim=-3, index=indices)
        v_select = torch.gather(v_, dim=-3, index=indices)
        o_inter = flash_attn_func(q, k_select, v_select)
        o_intra = flash_attn_func(q_, k_, v_, causal=True)
        o_intra = rearrange(o_intra, "b g c h d -> b (g c) h d")
        o = (
            g[:, :, :, 0].unsqueeze(-1) * o_inter
            + g[:, :, :, 1].unsqueeze(-1) * o_intra
        )

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        # outproj
        output = self.o_proj(output)

        return output, past_key_values
