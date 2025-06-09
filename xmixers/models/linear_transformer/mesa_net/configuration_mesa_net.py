# coding=utf-8
""" MesaNet configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MesaNetConfig(PretrainedConfig):
    model_type = "mesa_net"
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
        token_mixer_type="mesa_unit",
        embed_dim=1024,
        num_heads=8,
        bias=False,
        token_mixer_norm_type="rmsnorm",
        q_activation="silu",
        k_activation="silu",
        causal=True,
        gate_act="sigmoid",
        gate_pos="pre",
        threshold=0.99,
        lambda_initial_value=1.0,
        lambda_lower_bound=0.25,
        max_cg_step_training=30,
        max_cg_step_decoding=30,
        ##### channel mixer config
        channel_mixer_type="glu",
        mid_dim=1024,
        channel_mixer_activation="silu",
        use_gate_linear=True,
        ##### others
        max_position_embeddings=1024,
        use_output_gate=True,
        norm_type="rmsnorm",
        num_layers=12,
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
