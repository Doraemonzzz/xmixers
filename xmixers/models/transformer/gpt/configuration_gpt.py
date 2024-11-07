# coding=utf-8
""" GPT configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GPTConfig(PretrainedConfig):
    model_type = "gpt"
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
        base=10000,
        ape_type="sincos",
        # ffn config
        mid_dim=1024,
        ffn_activation="silu",
        # others
        num_layers=24,
        norm_type="layernorm",
        token_mixer_init_type=0,
        init_type=0,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        assert ape_type in ["sincos", "learnable"]
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
        self.base = base
        self.ape_type = ape_type
        # ffn config
        self.mid_dim = mid_dim
        self.ffn_activation = ffn_activation
        # others
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.token_mixer_init_type = token_mixer_init_type
        self.init_type = init_type
