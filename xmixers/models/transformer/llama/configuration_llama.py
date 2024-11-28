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
        ##### model config
        # attention config
        token_mixer_type="attn",
        embed_dim=1024,
        num_heads=8,
        kv_heads=-1,
        bias=False,
        use_lrpe=True,
        lrpe_type=1,
        base=10000,
        mpa_type=0,
        mpa_activation="none",
        # channel mixer config
        channel_mixer_type="glu",
        mid_dim=1024,
        channel_mixer_activation="silu",
        # for alu and lalu
        qk_dim=1024,
        v_dim=1024,
        mem_dim=1024,
        use_scale=0,
        use_output_gate=False,
        output_gate_activation="silu",
        use_low_rank_output_gate=False,
        channel_mixer_init_type=0,
        # others
        num_layers=24,
        norm_type="layernorm",
        token_mixer_init_type=0,
        init_type=0,
        rescale_type=0,
        use_postnorm=False,
        use_embed_scale=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        ##### hf origin
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.init_std = init_std
        ##### add
        # token mixer config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.bias = bias
        self.use_lrpe = use_lrpe
        self.lrpe_type = lrpe_type
        self.base = base
        self.token_mixer_type = token_mixer_type
        self.mpa_type = mpa_type
        self.mpa_activation = mpa_activation
        # channel mixer config
        self.channel_mixer_type = channel_mixer_type
        self.mid_dim = mid_dim
        self.channel_mixer_activation = channel_mixer_activation
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.mem_dim = mem_dim
        self.use_scale = use_scale
        self.use_output_gate = use_output_gate
        self.output_gate_activation = output_gate_activation
        self.use_low_rank_output_gate = use_low_rank_output_gate
        self.channel_mixer_init_type = channel_mixer_init_type
        # others
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.token_mixer_init_type = token_mixer_init_type
        self.init_type = init_type
        self.rescale_type = rescale_type
        self.use_postnorm = use_postnorm
        self.use_embed_scale = use_embed_scale
