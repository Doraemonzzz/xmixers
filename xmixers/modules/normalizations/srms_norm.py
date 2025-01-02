import torch
from xopes.ops.normalize import srmsnorm_fn


class SRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def forward(self, x, residual=None, return_residual=False):
        return srmsnorm_fn(
            x=x,
            dim=self.dim,
            eps=self.eps,
            residual=residual,
            return_residual=return_residual,
        )
