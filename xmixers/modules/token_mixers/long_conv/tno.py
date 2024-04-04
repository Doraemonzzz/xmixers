"""
Toeplitz Neural Operator in https://arxiv.org/pdf/2305.04749.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from xmixers.modules import get_norm_fn
from xmixers.ops import long_conv_1d_op
from xmixers.utils import XMIXERS_DEBUG, next_power_of_2, print_module, print_params

from .rpe import Rpe


class Tno(nn.Module):
    def __init__(
        self,
        in_dim: int,
        feature_dim: int,
        out_dim: int,
        activation: str = "silu",
        bias: bool = False,
        rpe_layers: int = 3,
        norm_type: str = "layernorm",
        use_decay: bool = True,
        causal: bool = True,
        dim: int = 1,
        lower_bound: float = 0.99,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.rpe = Rpe(
            in_dim=in_dim,
            feature_dim=feature_dim,
            out_dim=out_dim,
            activation=activation,
            bias=bias,
            rpe_layers=rpe_layers,
            norm_type=norm_type,
        )
        self.norm = get_norm_fn(norm_type)(out_dim)

        if use_decay:
            self.gamma = nn.Parameter(torch.randn(1, out_dim) * 0.1, requires_grad=True)
            # self.lower_bound = lower_bound
            # gamma = 1 / (torch.arange(1, out_dim + 1))
            # self.gamma = nn.Parameter(gamma.reshape(1, -1), requires_grad=True)

        self.use_decay = use_decay
        self.dim = dim
        self.zero = torch.empty(0)
        self.pos = torch.empty(0)
        self.neg = torch.empty(0)
        self.cache_size = 0
        self.causal = causal

    def extra_repr(self):
        return print_module(self)

    def get_w(self, x):
        n = x.shape[self.dim]
        m = next_power_of_2(n)
        if self.cache_size < n:
            self.cache_size = m
            self.zero = torch.zeros(1).unsqueeze(-1).to(x.device)
            self.pos = torch.arange(1, m).unsqueeze(-1).to(x.device)
            if not self.causal:
                self.neg = -torch.arange(1, m).unsqueeze(-1).flip(0).to(x.device)

        if self.causal:
            self.index = torch.cat([self.zero, self.pos[:n]], dim=0)
        else:
            self.index = torch.cat(
                [self.zero, self.pos[:n], self.zero, self.neg[-n:]], dim=0
            )

        return self.index

    def get_gamma(self, x):
        n = x.shape[self.dim]
        # gamma = self.lower_bound + (1 - self.lower_bound) * torch.clamp(self.gamma, min=0, max=1).float()
        # gamma_zero = torch.exp(self.zero * torch.log(self.gamma))
        # gamma_pos = torch.exp(self.pos[:n] * torch.log(self.gamma))

        gamma_zero = torch.exp(self.zero * F.logsigmoid(self.gamma))
        gamma_pos = torch.exp(self.pos[:n] * F.logsigmoid(self.gamma))

        if self.causal:
            gamma = torch.cat([gamma_zero, gamma_pos], dim=0)
        else:
            gamma = torch.cat(
                [gamma_zero, gamma_pos, gamma_zero, gamma_pos.flip(0)], dim=0
            )
        # print(gamma)
        return gamma

    def forward(self, x):
        index = self.get_w(x)
        w = self.rpe(index).to(torch.float32)
        if self.use_decay:
            w = self.get_gamma(x) * w
        y = self.norm(long_conv_1d_op(x, w, self.dim))

        return y
