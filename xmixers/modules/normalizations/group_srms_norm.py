from typing import List, Union

import torch
from torch import Size

_shape_t = Union[int, List[int], Size]


from xopes.ops.normalize import group_srms_norm_fn


class GroupSRMSNorm(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        eps: float = 1e-5,
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

    def forward(self, x, residual=None, return_residual=False):
        return group_srms_norm_fn(
            x=x,
            dim=x.shape[-1],
            eps=self.eps,
            residual=residual,
            return_residual=return_residual,
            num_groups=self.num_groups,
        )

    def extra_repr(self) -> str:
        return "num_groups={num_groups}, num_channels={num_channels}, eps={eps}".format(
            **self.__dict__
        )
