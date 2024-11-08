import torch
import torch.nn as nn
from einops import pack


class MlpPe(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        base: int = 10000,
        bias: bool = False,
    ):
        super().__init__()
        theta = (
            base
            ** (
                -2 / embed_dim * torch.arange(embed_dim // 2, dtype=torch.int64)
            ).float()
        )
        self.linear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.register_buffer("theta", theta, persistent=False)
        self.embed_dim = embed_dim
        self.pe = torch.empty(0)

    def get_pe(self, x, shape=None):
        # x: b, ... , d
        # compute index
        if shape is None:
            shape = x.shape[1:-1]
        m = len(shape)
        array = [
            torch.arange(n, dtype=torch.int64, device=torch.cuda.current_device())
            for n in shape
        ]
        grid = torch.meshgrid(array)
        index = torch.stack(grid, dim=-1)

        # compute theta
        d = self.embed_dim // 2 // m

        theta = []
        for i in range(m):
            theta.append(index[..., i : i + 1] * self.theta[:d])

        theta = torch.cat(theta, dim=-1)

        # compute pe
        pe = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)

        if len(x.shape) == 3:
            # b, n, d case
            pe, ps = pack([pe], "* d")

        self.pe = pe

    def forward(self, x, shape=None, offset=0):
        if self.pe.shape[0] == 0 or self.pe.shape[0] < (offset + x.shape[-2]):
            assert len(x.shape) == 3, "only support 1d case"
            self.get_pe(x, [offset + x.shape[-2]])
        start = offset
        end = offset + x.shape[1]
        pe = self.linear(
            self.pe[
                start:end,
            ]
        )
        x = x + pe

        return x
