# coding=utf-8
""" ImplicitLinearTransformer configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ImplicitLinearTransformerConfig(PretrainedConfig):
    model_type = "implicit_linear_transformer"
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
        embed_dim=1024,
        num_heads=8,
        bias=False,
        scalar_decay=False,
        token_mixer_type="implicit_value_attn",
        gate_act="sigmoid",
        gate_pos="pre",
        token_mixer_norm_type="rmsnorm",
        use_decay=True,
        ##### channel mixer config
        mid_dim=1024,
        channel_mixer_type="glu",
        channel_mixer_activation="silu",
        use_gate_linear=True,
        ##### others
        max_position_embeddings=1024,
        num_layers=24,
        use_output_gate=True,
        norm_type="rmsnorm",
        q_activation="silu",
        k_activation="silu",
        threshold=0.99,
        use_offset=False,
        causal=True,
        use_embed_scale=False,
        ce_type="xopes_flce",
        pad_embed_dim=True,
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
        if scalar_decay:
            share_decay = False
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
