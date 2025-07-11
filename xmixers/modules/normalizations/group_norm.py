from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Size

_shape_t = Union[int, List[int], Size]


class GroupNorm(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        eps: float = 1e-5,
        affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        **kwargs
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, x, residual=None, return_residual=False):
        return group_norm_fn(
            x=x,
            weight=self.weight,
            bias=self.bias,
            dim=x.shape[-1],
            eps=self.eps,
            residual=residual,
            return_residual=return_residual,
            num_groups=self.num_groups,
        )

    def extra_repr(self) -> str:
        return "num_groups={num_groups}, num_channels={num_channels}, eps={eps}, affine={affine}".format(
            **self.__dict__
        )
