# coding=utf-8
""" Hgrn3 configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Hgrn3Config(PretrainedConfig):
    model_type = "hgrn3"
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
        expand_ratio=8,
        bias=False,
        # glu config
        mid_dim=1024,
        glu_activation="silu",
        # others
        num_layers=24,
        use_output_gate=True,
        norm_type="layernorm",
        q_activation="silu",
        k_activation="silu",
        beta_activation="silu",
        causal=True,
        use_dense_memory=True,
        n_min=2,
        n_max=256,
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
        self.expand_ratio = expand_ratio
        self.bias = bias
        # glu config
        self.mid_dim = mid_dim
        self.glu_activation = glu_activation
        # others
        self.num_layers = num_layers
        self.use_output_gate = use_output_gate
        self.norm_type = norm_type
        self.q_activation = q_activation
        self.k_activation = k_activation
        self.beta_activation = beta_activation
        self.causal = causal
        self.use_dense_memory = use_dense_memory
        self.n_min = n_min
        self.n_max = n_max
