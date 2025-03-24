import torch.nn as nn
import torch.nn.functional as F

from .utils import round_ste


class FiniteScalarQuantizer(nn.Module):
    def __init__(
        self,
        num_bins=128,
        center=False,
        **kwargs,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.center = center

    def forward(self, x):
        x_quant = round_ste(F.sigmoid(x))
        if self.center:
            x_quant = 2 * x_quant - 1

        return x_quant
