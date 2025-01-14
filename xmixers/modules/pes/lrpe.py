"""
Lrpe in https://openreview.net/forum?id=xoLyps2qWc
"""

import numpy as np
import torch
import torch.nn as nn

from xmixers.utils import XMIXERS_DEBUG, logging_info, print_params


class Lrpe(nn.Module):
    def __init__(
        self,
        head_dim: int = 128,
        num_heads: int = 8,
        lrpe_type: int = 1,
        base: int = 10000,
        max_position_embeddings: int = 1024,
    ):
        """
        lrpe_type: 1 for standard rope, 2 for mix rope (rope half head dim), 3 for complex version(cosformer style)
        """
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.lrpe_type = lrpe_type
        self.base = base
        self.index = torch.empty(0)

        d = self.num_heads * self.head_dim
        if self.lrpe_type == 1:
            logging_info("lrpe rope")
            theta = base ** (
                -2 / d * torch.arange(d // 2, dtype=torch.int64)
            ).float().reshape(num_heads, 1, -1)
            self.register_buffer("theta", theta, persistent=False)
        elif lrpe_type == 2:
            logging_info("lrpe mix rope")
            theta = base ** (
                -2 / d * torch.arange(d // 2 // 2, dtype=torch.int64)
            ).float().reshape(num_heads, 1, -1)
            self.register_buffer("theta", theta, persistent=False)
        elif lrpe_type == 3:
            logging_info("lrpe complex transform")
            theta = base ** (
                -2 / d * torch.arange(d, dtype=torch.int64)
            ).float().reshape(num_heads, 1, -1)
            self.theta = nn.Parameter(theta)
        elif lrpe_type == 4:
            logging_info("cosformer transform")
            theta = (
                torch.tensor(np.pi / 2 / max_position_embeddings)
                .float()
                .reshape(1, 1, -1)
            )
            self.theta = nn.Parameter(theta)
        elif lrpe_type == 5:
            logging_info("share rope, standard version")
            d = head_dim
            theta = base ** (
                -2 / d * torch.arange(d // 2, dtype=torch.int64)
            ).float().reshape(1, 1, -1)
            self.register_buffer("theta", theta, persistent=False)
        elif lrpe_type == 6:
            logging_info("share rope, not standard version")
            d = head_dim
            theta = base ** (
                -2 / (d * num_heads) * torch.arange(d // 2, dtype=torch.int64)
            ).float().reshape(1, 1, -1)
            self.register_buffer("theta", theta, persistent=False)
        elif lrpe_type == 7:
            logging_info("lrpe complex transform, theta very small")
            theta = base ** (
                -2 / head_dim * torch.arange(d, dtype=torch.int64)
            ).float().reshape(num_heads, 1, -1)
            self.theta = nn.Parameter(theta)
        elif lrpe_type == 8:
            logging_info("lrpe rope, theta very small")
            theta = base ** (
                -2 / head_dim * torch.arange(d // 2, dtype=torch.int64)
            ).float().reshape(num_heads, 1, -1)
            self.register_buffer("theta", theta, persistent=False)
        elif lrpe_type == 9:
            logging_info("lrpe mix rope, theta very small")
            theta = base ** (
                -2 / head_dim * torch.arange(d // 2 // 2, dtype=torch.int64)
            ).float().reshape(num_heads, 1, -1)
            self.register_buffer("theta", theta, persistent=False)
        elif lrpe_type == 10:
            logging_info("lrpe complex transform, head share")
            theta = base ** (
                -2 / head_dim * torch.arange(head_dim, dtype=torch.int64)
            ).float().reshape(1, 1, -1)
            self.theta = nn.Parameter(theta)
        else:
            raise ValueError(f"lrpe_type: {lrpe_type} has not been support!")

    def init_weights(self):
        d = self.num_heads * self.head_dim
        if self.lrpe_type == 1:
            with torch.device(self.theta.device):
                theta = self.base ** (
                    -2 / d * torch.arange(d // 2, dtype=torch.int64)
                ).float().reshape(self.num_heads, 1, -1)
                self.theta = theta
        elif lrpe_type == 2:
            with torch.device(self.theta.device):
                theta = self.base ** (
                    -2 / d * torch.arange(d // 2 // 2, dtype=torch.int64)
                ).float().reshape(self.num_heads, 1, -1)
                self.theta = theta
        elif lrpe_type == 3:
            with torch.device(self.theta.device):
                theta = self.base ** (
                    -2 / d * torch.arange(d // 2, dtype=torch.int64)
                ).float().reshape(self.num_heads, 1, -1)
                self.theta.data = theta

    def forward(self, x, offset=0):
        n, d = x.shape[-2], x.shape[-1]

        if self.index.shape[0] == 0 or self.index.shape[1] < (offset + n):
            self.index = (
                torch.arange(n, dtype=torch.int64)
                .reshape(1, -1, 1)
                .to(self.theta.device)
            )

        if self.lrpe_type in [1, 5, 6, 8]:
            theta = (self.index[:, :n] + offset) * self.theta
            theta_ = torch.polar(
                torch.ones_like(theta).to(torch.float32), theta.to(torch.float32)
            )
            x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            x_out = torch.view_as_real(x_ * theta_).flatten(3).type_as(x)
        elif self.lrpe_type in [2, 9]:
            index = self.index[:, :n] + offset
            # only support even number
            e = d // 2 + d % 2
            # last e features
            x1 = x[..., e:]
            # do rope for the first e features
            x = x[..., :e]
            # get theta
            theta = self.theta
            theta = torch.stack([theta, theta], dim=-1).reshape(-1, 1, e)
            theta = theta * index
            # (-q1, -q3), (q0, q2) -> (-q1, q0, -q3, q2)
            x_half = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
            x_transform = x * torch.cos(theta) + x_half * torch.sin(theta)
            x_out = torch.cat([x_transform, x1], dim=-1).to(x.dtype)
        elif self.lrpe_type in [3, 4, 7, 10]:
            index = self.index[:, :n] + offset
            theta = self.theta.float() * index
            x_out = torch.concat(
                [x * torch.cos(theta), x * torch.sin(theta)], dim=-1
            ).to(x.dtype)

        return x_out
