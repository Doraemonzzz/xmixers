import torch
import torch.nn.functional as F
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

# credit to: https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/attn.py
def _upad_input(q, k, v, attention_mask, q_len):
    seqlens = attention_mask.sum(-1, dtype=torch.int32)
    indices_k = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_k = seqlens.max().item()
    cu_seqlens_k = F.pad(cumsum_fn(seqlens, dim=0), (1, 0))

    num_heads = q.shape[-2]
    batch_size, seq_len, num_key_value_heads, head_dim = k.shape

    k = index_first_axis(
        k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k
    )
    v = index_first_axis(
        v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k
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
        k,
        v,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_q, max_seqlen_k),
    )
