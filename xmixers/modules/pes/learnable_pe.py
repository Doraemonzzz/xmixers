import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePe(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_sequence_length: int = 2048,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embed_dim = embed_dim
        weight = torch.randn(max_sequence_length, embed_dim)
        # weight = self.sin_cos_pe(max_sequence_length, embed_dim)
        self.weight = nn.Parameter(weight, requires_grad=True)

    # def sin_cos_pe(self, max_sequence_length, embed_dim):
    #     base = 10000
    #     theta = (
    #         base
    #         ** (
    #             -2 / embed_dim * torch.arange(embed_dim // 2, dtype=torch.int64)
    #         ).float()
    #     )
    #     index = torch.arange(max_sequence_length, dtype=torch.int64)
    #     theta = torch.einsum("n, d -> n d", index, theta)
    #     weight = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)

    #     return weight

    def extra_repr(self) -> str:
        s = "{max_sequence_length}, {embed_dim}"
        return s.format(**self.__dict__)

    def forward(self, x, shape=None, offset=0):
        n = x.shape[1]
        pos = torch.arange(0, n, dtype=torch.long, device=x.device) + offset
        pe = F.embedding(pos, self.weight)
        x = x + pe

        return x
