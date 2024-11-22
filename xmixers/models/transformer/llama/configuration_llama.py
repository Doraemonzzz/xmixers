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
        embed_dim=1024,
        num_heads=8,
        kv_heads=-1,
        bias=False,
        use_lrpe=True,
        lrpe_type=1,
        base=10000,
        # glu config
        mid_dim=1024,
        glu_activation="silu",
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
        # attention config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.bias = bias
        self.use_lrpe = use_lrpe
        self.lrpe_type = lrpe_type
        self.base = base
        # glu config
        self.mid_dim = mid_dim
        self.glu_activation = glu_activation
        # others
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.token_mixer_init_type = token_mixer_init_type
        self.init_type = init_type
        self.rescale_type = rescale_type
        self.use_postnorm = use_postnorm
        self.use_embed_scale = use_embed_scale
