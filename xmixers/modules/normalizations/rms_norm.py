"""
SimpleRMSNorm in https://arxiv.org/abs/2307.14995
RMSNorm in https://arxiv.org/pdf/1910.07467.pdf
GatedRMSNorm in https://arxiv.org/pdf/2104.07012.pdf

Reference:
https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
https://github.com/bzhangGo/zero/blob/master/modules/rela.py
"""
import torch
import torch.nn as nn

from xmixers.modules import BaseModule


class SimpleRMSNorm(BaseModule):
    def __init__(self, d: int, eps: float = 1e-8) -> None:
        super(SimpleRMSNorm, self).__init__()
        self.eps = eps
        self.d = d

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        return x_normed


class RMSNorm(BaseModule):
    def __init__(
        self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False
    ) -> None:
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class GatedRMSNorm(BaseModule):
    def __init__(self, d: int, eps: float = 1e-8, bias: bool = False) -> None:
        super(GatedRMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        self.gate = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        return self.scale * x_normed * torch.sigmoid(self.gate * x)
