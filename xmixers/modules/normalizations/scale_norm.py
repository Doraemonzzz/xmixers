import torch
from torch import nn


class ScaleNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.d = d
        self.eps = eps
        self.scala = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean_square = (x**2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps) * self.scala
        return x
