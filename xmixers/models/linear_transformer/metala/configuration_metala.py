# coding=utf-8
""" MetaLa configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class MetaLaConfig(PretrainedConfig):
    model_type = "metala"
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
        max_position_embeddings=1024,
        num_layers=24,
        use_output_gate=False,
        non_sparse_ratio=1,
        num_sparse=4,
        norm_type="layernorm",
        q_activation="silu",
        causal=True,
        use_embed_scale=False,
        pad_embed_dim=True,
        # init
        init_type=0,
        token_mixer_init_type=0,
        rescale_type=0,
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
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.use_output_gate = use_output_gate
        self.non_sparse_ratio = non_sparse_ratio
        self.num_sparse = num_sparse
        self.norm_type = norm_type
        self.q_activation = q_activation
        self.causal = causal
        self.use_embed_scale = use_embed_scale
        # init
        self.init_type = init_type
        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.channel_mixer_init_type = channel_mixer_init_type
