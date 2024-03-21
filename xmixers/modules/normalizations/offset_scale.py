import torch
import torch.nn as nn


class OffsetScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = torch.einsum("... d, d -> ... d", x, self.gamma) + self.beta
        return out
