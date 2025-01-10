import torch.nn as nn

from .group_norm import GroupNorm
from .group_rms_norm import GroupRMSNorm
from .group_srms_norm import GroupSRMSNorm
from .l2_norm import l2_norm
from .layer_norm import LayerNorm
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
    elif norm_type == "groupnorm":
        return GroupNorm
    elif norm_type == "grouprmsnorm":
        return GroupRMSNorm
    elif norm_type == "groupsrmsnorm":
        return GroupSRMSNorm
    else:
        return LayerNorm
