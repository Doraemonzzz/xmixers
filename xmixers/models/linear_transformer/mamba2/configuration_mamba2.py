# coding=utf-8
""" Mamba2 configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Mamba2XmixersConfig(PretrainedConfig):
    model_type = "mamba2_"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        vocab_size=64000,
        use_cache=True,
        init_std=0.02,
        tie_word_embeddings=False,
        ########## model config
        ##### token mixer config
        token_mixer_type="mamba2",
        embed_dim=1024,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        layer_idx=0,
        token_mixer_norm_type="rmsnorm",
        gate_act="sigmoid",
        gate_pos="pre",
        use_lightning=False,
        ##### others
        max_position_embeddings=1024,
        num_layers=24,
        norm_type="rmsnorm",
        causal=True,
        use_embed_scale=False,
        ce_type="xopes_flce",
        ##### init
        init_type=1,
        token_mixer_init_type=4,
        rescale_type=2,
        gain=0.01,
        channel_mixer_init_type=0,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        for key, value in locals().items():
            if key not in [
                "self",
                "kwargs",
                "__class__",
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "tie_word_embeddings",
            ]:
                setattr(self, key, value)
