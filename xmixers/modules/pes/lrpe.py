"""
Lrpe in https://openreview.net/forum?id=xoLyps2qWc
"""

from typing import Optional

import torch
import torch.nn as nn
from xopes.ops import lrpe_fn

from xmixers.utils import XMIXERS_DEBUG, logging_info, print_params


class Lrpe(nn.Module):
    def __init__(
        self,
        head_dim: int = 128,
        num_heads: int = 8,
        lrpe_type: int = 1,
        base: int = 10000,
        act: str = "none",
        act_dim: Optional[int] = None,
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
        self.base = base
        self.act = act
        self.act_dim = act_dim

        d = self.head_dim
        if lrpe_type in [1, 2, 3]:
            self.lrpe_type = "rotate"
        elif lrpe_type in [4, 5, 6]:
            self.lrpe_type = "cosine"
        else:
            raise ValueError(f"lrpe_type: {lrpe_type} has not been support!")

        # init parameters
        self.register_buffer("theta", torch.empty(0), persistent=False)
        self.lrpe_type_ = lrpe_type
        self.base = base
        self.d = d
        self._init_weights()

    def _init_weights(self):
        lrpe_type = self.lrpe_type_
        base = self.base
        d = self.d
        if lrpe_type == 1:
            logging_info("lrpe rotate, i.e, rope")
            theta = base ** (
                -2 / d * torch.arange(d // 2, dtype=torch.int64)
            ).float().reshape(1, -1)
        elif lrpe_type == 2:  # result much worse than 3
            logging_info("lrpe mix rotate, rotate half head dim, use low freq")
            theta = (
                base
                ** (-2 / d * torch.arange(d // 2, dtype=torch.int64))
                .float()
                .reshape(1, -1)[:, d // 4 :]
            )
            theta = torch.cat([torch.zeros_like(theta), theta], dim=-1)
        elif lrpe_type == 3:
            logging_info("lrpe mix rotate, rotate half head dim, use high freq")
            theta = (
                base
                ** (-2 / d * torch.arange(d // 2, dtype=torch.int64))
                .float()
                .reshape(1, -1)[:, : d // 4]
            )
            theta = torch.cat([theta, torch.zeros_like(theta)], dim=-1)
        elif lrpe_type == 4:
            logging_info("lrpe cosine")
            theta = base ** (
                -2 / d * torch.arange(d, dtype=torch.int64)
            ).float().reshape(1, -1)
        elif lrpe_type == 5:  # result much worse than 6
            logging_info("lrpe cosine, cosine half head dim, use low freq")
            theta = (
                base
                ** (-2 / d * torch.arange(d, dtype=torch.int64))
                .float()
                .reshape(1, -1)[:, d // 2 :]
            )
            theta = torch.cat([torch.zeros_like(theta), theta], dim=-1)
        elif lrpe_type == 6:
            logging_info("lrpe cosine, cosine half head dim, use high freq")
            theta = (
                base
                ** (-2 / d * torch.arange(d, dtype=torch.int64))
                .float()
                .reshape(1, -1)[:, : d // 2]
            )
            theta = torch.cat(
                [
                    theta,
                    torch.zeros_like(theta),
                ],
                dim=-1,
            )
        self.theta = theta.to(self.theta.device)
        self._is_hf_initialized = True

    def forward(self, x, offset=0):
        return lrpe_fn(
            x=x,
            theta=self.theta,
            offset=offset,
            act=self.act,
            dim=self.act_dim,
            lrpe_type=self.lrpe_type,
        )
