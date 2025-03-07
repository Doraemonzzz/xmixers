# coding=utf-8
""" LLaMA configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LLaMAConfig(PretrainedConfig):
    model_type = "llama_"
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
        token_mixer_type="attn",
        embed_dim=1024,
        num_heads=8,
        kv_heads=-1,
        bias=False,
        use_lrpe=True,
        lrpe_type=1,
        base=10000,
        mpa_type=0,
        mpa_activation="sigmoid",
        use_l2_norm=False,
        gate_type=0,
        head_dim=-1,
        q_rank=8,
        kv_rank=2,
        cp_activation="none",
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        window_size=-1,
        chunk_size=128,
        token_mixer_top_k=2,
        ###### channel mixer config
        channel_mixer_type="glu",
        mid_dim=1024,
        channel_mixer_activation="silu",
        use_gate_linear=True,
        # for alu and lalu
        qk_dim=1024,
        v_dim=1024,
        mem_dim=1024,
        use_scale=0,
        use_output_gate=False,
        output_gate_activation="silu",
        use_low_rank_output_gate=False,
        channel_mixer_init_type=0,
        ##### others
        max_position_embeddings=1024,
        num_layers=24,
        norm_type="rmsnorm",
        use_postnorm=False,
        use_embed_scale=False,
        ce_type="xopes_flce",
        fuse_norm_add=False,
        ##### init
        init_type=1,
        token_mixer_init_type=4,
        rescale_type=2,
        gain=0.01,
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
