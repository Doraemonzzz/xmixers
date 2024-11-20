# nGLU: https://arxiv.org/pdf/2410.01131

import torch
import torch.nn as nn

from xmixers.modules.activations import get_activation_fn
from xmixers.utils import XMIXERS_DEBUG, print_module, print_params


class nGLU(nn.Module):
    def __init__(
        self, embed_dim: int, mid_dim: int, activation: str, bias: bool = False
    ) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.w1 = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.w2 = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.w3 = nn.Linear(mid_dim, embed_dim, bias=bias)
        self.act = get_activation_fn(activation)
        self.embed_dim = embed_dim

        self.suv_init_value = 1.0
        self.suv_init_scaling = 1.0
        self.su = torch.nn.Parameter(
            self.suv_init_scaling * torch.ones(mid_dim, dtype=torch.float32)
        )
        self.sv = torch.nn.Parameter(
            self.suv_init_scaling * torch.ones(mid_dim, dtype=torch.float32)
        )

    def extra_repr(self):
        return print_module(self)

    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, x):
        v = self.w1(x)
        u = self.w2(x)
        v = (
            self.sv
            * ((self.suv_init_value / self.suv_init_scaling) * (self.embed_dim**0.5))
        ) * v
        u = (
            self.su
            * ((self.suv_init_value / self.suv_init_scaling) * (self.embed_dim**0.5))
        ) * u
        output = self.w3(u * self.act(v))

        return output
