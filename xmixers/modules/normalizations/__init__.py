from .offset_scale import *
from .rms_norm import *
from .scale_norm import *
from .srmsnorm import SRMSNorm


def get_norm_fn(norm_type: str) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm
    elif norm_type == "gatedrmsnorm":
        return GatedRMSNorm
    elif norm_type == "srmsnorm":
        return SRMSNorm
    elif norm_type == "scalenorm":
        return ScaleNorm
    else:
        return nn.LayerNorm
