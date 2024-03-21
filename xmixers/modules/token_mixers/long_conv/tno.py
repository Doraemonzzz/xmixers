import torch
import torch.nn as nn
import torch.nn.functional as F

from xmixers.ops import long_conv_1d_op
from xmixers.utils import next_power_of_2, print_module, print_params

from .rpe import Rpe


class Tno(nn.Module):
    def __init__(
        self,
        in_dim,
        feature_dim,
        out_dim,
        activation="silu",
        bias=False,
        rpe_layers=3,
        norm_type="layernorm",
        use_decay=True,
        causal=True,
        dim=1,
        **kwargs,
    ):
        super().__init__()
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

        if use_decay:
            self.gamma = nn.Parameter(torch.randn(1, out_dim) * 0.1, requires_grad=True)

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
        gamma_zero = torch.exp(self.zero * F.logsigmoid(self.gamma))
        gamma_pos = torch.exp(self.pos[:n] * F.logsigmoid(self.gamma))

        if self.causal:
            gamma = torch.cat([gamma_zero, gamma_pos], dim=0)
        else:
            gamma = torch.cat(
                [gamma_zero, gamma_pos, gamma_zero, gamma_pos.flip(0)], dim=0
            )

        return gamma

    def forward(self, x):
        index = self.get_w(x)
        w = self.rpe(index)
        if self.use_decay:
            w = self.get_gamma(x) * w
        y = long_conv_1d_op(x, w, self.dim)

        return y
