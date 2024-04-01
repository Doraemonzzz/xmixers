from torch import nn

from xmixers.modules.normalizations import (
    GatedRMSNorm,
    RMSNorm,
    ScaleNorm,
    SimpleRMSNorm,
)


def get_norm_fn(norm_type: str) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm
    elif norm_type == "gatedrmsnorm":
        return GatedRMSNorm
    elif norm_type == "simplermsnorm":
        return SimpleRMSNorm
    elif norm_type == "scalenorm":
        return ScaleNorm
    else:
        return nn.LayerNorm
