import torch
import torch.nn as nn

TORCH_VERSION = torch.__version__

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_heads: int = -1,
        use_lrpe: bool = True,
        bias: bool = False,
        layer_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.kv_heads = kv_heads
        self.head_dim = hidden_dim // num_heads
        if self.kv_heads == -1:
            self.qkv_proj = nn.Linear(embed_dim, 3 * hidden_dim, bias=bias)
        else:
            d = self.kv_heads * self.head_dim
            self.qkv_proj = nn.Linear(embed_dim, hidden_dim + 2 * d, bias=bias)
        self.out_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.num_heads = num_heads

        self.use_lrpe = use_lrpe

        if self.use_lrpe:
            self.lrpe = Lrpe(
                num_heads=self.num_heads,
                embed_dim=self.head_dim,
            )

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        output_attentions: bool = False,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        # x: b n d
        x.shape[-2]

        # linear map
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", h=self.num_heads),
            [q, k, v],
        )

        # lrpe
        q_offset = 0
        if past_key_values is not None:
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k)

        # cache update
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        q, k, v = map(lambda x: rearrange(x, "... h n d -> ... n h d"), [q, k, v])

        if attention_mask is None:
            output = flash_attn_func(q, k, v, causal=True)
        else:
            assert False, "flash_attn_varlen_qkvpacked_func current not support"

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        # outproj
        output = self.out_proj(output)

        return output, past_key_values
