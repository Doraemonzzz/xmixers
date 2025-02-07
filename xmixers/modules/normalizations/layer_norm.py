import numbers
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Size

_shape_t = Union[int, List[int], Size]

from xopes.ops.normalize import layer_norm_fn


class LayerNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, x, residual=None, return_residual=False):
        return layer_norm_fn(
            x=x,
            weight=self.weight,
            bias=self.bias,
            dim=x.shape[-1],
            eps=self.eps,
            residual=residual,
            return_residual=return_residual,
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
