# coding=utf-8
""" LinearTransformer configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LinearTransformerConfig(PretrainedConfig):
    model_type = "linear_transformer"
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
        use_output_gate=True,
        norm_type="layernorm",
        linear_activation="silu",
        causal=True,
        use_ape=False,
        use_dense_memory=False,
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
        self.use_output_gate = use_output_gate
        self.linear_activation = linear_activation
        self.causal = causal
        # glu config
        self.mid_dim = mid_dim
        self.glu_activation = glu_activation
        # others
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.use_ape = use_ape
        self.use_dense_memory = use_dense_memory
        self.token_mixer_init_type = token_mixer_init_type
        self.init_type = init_type
