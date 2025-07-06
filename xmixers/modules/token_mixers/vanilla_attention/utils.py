from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from xopes.ops import cumsum_fn

try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except:
    flash_attn_func = None
    index_first_axis = None
    pad_input = None
    unpad_input = None

_pad_input = pad_input

# credit to: https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/utils.py
def _unpad_input(
    q: torch.Tensor,
    states: Tuple[torch.Tensor],
    attention_mask: torch.Tensor,
    q_len: int,
    keepdim: bool = False,
):
    seqlens = attention_mask.sum(-1, dtype=torch.int32)
    indices_k = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_k = seqlens.max().item()
    cu_seqlens_k = F.pad(cumsum_fn(seqlens, dim=0), (1, 0))

    num_heads = q.shape[-2]
    batch_size, seq_len, num_key_value_heads, head_dim = states[0].shape

    states = (
        index_first_axis(rearrange(state, "b s ... -> (b s) ..."), indices_k)
        for state in states
    )

    if q_len == seq_len:
        q = index_first_axis(
            q.reshape(batch_size * seq_len, num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_q = max_seqlen_k
        indices_q = indices_k
    elif q_len == 1:
        max_seqlen_q = 1
        # There is a memcpy here, that is very bad.
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
        indices_q = cu_seqlens_q[:-1]
        q = q.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -q_len:]
        q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

    return (
        q,
        states,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_q, max_seqlen_k),
    )
