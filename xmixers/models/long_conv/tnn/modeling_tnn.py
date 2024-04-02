from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from xmixers.modules import GLU, BaseModule, Gtu, get_norm_fn

from .configuration_tnn import TnnConfig

logger = logging.get_logger(__name__)


class TnnLayer(BaseModule):
    def __init__(
        self,
        embed_dim: int,
        expand_ratio: int = 1,
        bias: bool = False,
        gtu_activation: str = "silu",
        causal: bool = True,
        norm_type: str = "layernorm",
        use_decay: bool = True,
        in_dim: int = 1,
        feature_dim: int = 32,
        rpe_layers: int = 3,
        dims: List[int] = [-2],
        # glu config
        mid_dim: int = 1024,
        glu_activation: str = "silu",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.token_mixer = Gtu(
            embed_dim=embed_dim,
            expand_ratio=expand_ratio,
            bias=bias,
            activation=gtu_activation,
            causal=causal,
            norm_type=norm_type,
            use_decay=use_decay,
            in_dim=in_dim,
            feature_dim=feature_dim,
            rpe_layers=rpe_layers,
            dims=dims,
        )

        self.token_norm = get_norm_fn(norm_type)(embed_dim)

        self.channel_mixer = GLU(
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            activation=glu_activation,
            bias=bias,
        )

        self.channel_norm = get_norm_fn(norm_type)(embed_dim)

    def forward(
        self,
        x,
        attn_mask: Optional[torch.Tensor] = None,
        attn_padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        x = self.token_mixer(self.token_norm(x)) + x
        x = self.channel_mixer(self.channel_norm(x)) + x

        outputs = (x,)

        # add this later
        attn_weights, present_key_value = None, None

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class TnnPreTrainedModel(PreTrainedModel):
    config_class = TnnConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["TnnLayer"]

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, TransnormerModel):
    #         module.gradient_checkpointing = value
