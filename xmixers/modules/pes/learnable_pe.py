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
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self) -> str:
        s = "{max_sequence_length}, {embed_dim}"
        return s.format(**self.__dict__)

    def forward(self, x, shape=None, offset=0):
        n = x.shape[1]
        pos = torch.arange(0, n, dtype=torch.long, device=x.device) + offset
        pe = F.embedding(pos, self.weight)
        x = x + pe

        return x
