# coding=utf-8
""" FlexGPT configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class FlexGPTConfig(PretrainedConfig):
    model_type = "flex_gpt"
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
        token_mixer_type="flex_attn",
        embed_dim=1024,
        num_heads=8,
        kv_heads=-1,
        bias=False,
        window_size=-1,
        token_mixer_norm_type="grouprmsnorm",
        ###### channel mixer config
        channel_mixer_type="glu",
        mid_dim=1024,
        channel_mixer_activation="silu",
        use_gate_linear=True,
        ##### others
        max_position_embeddings=1024,
        num_layers=24,
        norm_type="rmsnorm",
        use_embed_scale=False,
        ce_type="xopes_flce",
        fuse_norm_add=False,
        rpe_type=0,  # 0: no rpe, 1: alibi
        n_min=2,
        n_max=256,
        ##### init
        init_type=1,
        token_mixer_init_type=4,
        rescale_type=2,
        gain=0.01,
        pad_embed_dim=True,
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
