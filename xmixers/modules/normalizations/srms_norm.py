import torch
import torch.nn.functional as F
from xopes.ops.normalize import srms_norm_fn


class SRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def forward(self, x, residual=None, return_residual=False):
        # for DTensor, we don't use residual
        if isinstance(x, torch.distributed.tensor.DTensor):
            return F.rms_norm(input=x, normalized_shape=(self.dim,), eps=self.eps)
        else:
            return srms_norm_fn(
                x=x,
                dim=self.dim,
                eps=self.eps,
                residual=residual,
                return_residual=return_residual,
            )
