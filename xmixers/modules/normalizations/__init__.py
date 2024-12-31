import torch.nn as nn

from .l2_norm import l2_norm
from .offset_scale import OffsetScale
from .rms_norm import GatedRMSNorm, RMSNorm
from .scale_norm import ScaleNorm
from .srms_norm import SRMSNorm


def get_norm_fn(norm_type: str):
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
