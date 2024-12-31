import torch
from xopes.ops.normalize import normalize_fn


def srmsnorm_fn(x, dim, eps=1e-6, residual=None):
    return normalize_fn(
        x=x,
        weight=None,
        bias=None,
        residual=residual,
        c=dim**0.5,
        eps=eps,
        use_mean=False,
        num_groups=1,
    )


class SRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def forward(self, x, residual=None):
        return srmsnorm_fn(
            x=x,
            dim=self.dim,
            eps=self.eps,
            residual=residual,
        )
