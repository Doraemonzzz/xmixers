from typing import Optional

import torch
import torch.nn as nn
from xopes.ops.normalize import normalize_fn


class NormOp(nn.Module):
    def __init__(
        self,
        norm_type: str,
    ):
        self.norm_type = norm_type
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        dim: float = 1.0,
        eps: float = 1e-6,
        use_mean: bool = False,
        num_groups: int = 1,
        return_residual: bool = False,
    ):
        if self.norm_type == "layernorm":
            c = dim**0.5
            use_mean = True
            num_groups = 1
        elif self.norm_type == "rmsnorm":
            bias = None
            c = dim**0.5
            use_mean = False
            num_groups = 1
        elif self.norm_type == "srmsnorm":
            weight = None
            bias = None
            c = dim**0.5
            use_mean = False
            num_groups = 1
        elif self.norm_type == "groupnorm":
            group_size = dim // num_groups
            c = group_size**0.5
            use_mean = True
        elif self.norm_type == "grouprmsnorm":
            group_size = dim // num_groups
            c = group_size**0.5
            use_mean = False
        elif self.norm_type == "groupsrmsnorm":
            group_size = dim // num_groups
            c = group_size**0.5
            use_mean = False
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")

        return normalize_fn(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            c=c,
            eps=eps,
            use_mean=use_mean,
            num_groups=num_groups,
            return_residual=return_residual,
        )
