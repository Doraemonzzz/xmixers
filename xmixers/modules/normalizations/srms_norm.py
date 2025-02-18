import torch

from .utils import NormOp


class SRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.op = NormOp(norm_type="srmsnorm")

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def forward(self, x, residual=None, return_residual=False):
        return self.op(
            x,
            None,
            None,
            residual,
            self.dim,
            self.eps,
            False,
            1,
            return_residual,
        )
