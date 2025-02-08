# coding=utf-8
""" ChunkRnn configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ChunkRnnConfig(PretrainedConfig):
    model_type = "chunk_rnn"
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
        token_mixer_norm_type="layernorm",
        token_mixer_type="chunk_rnn",
        # chunk params
        chunk_type: int = 0,
        gradient_type: int = 0,
        use_init_weights: bool = False,
        use_scale: bool = False,
        chunk_size: int = 128,
        # lrpe
        use_lrpe: bool = True,
        lrpe_type: int = 1,
        base: int = 10000,
        # glu config
        mid_dim=1024,
        channel_mixer_type="glu",
        channel_mixer_activation="silu",
        use_gate_linear=True,
        # others
        max_position_embeddings=1024,
        num_layers=24,
        use_output_gate=False,
        norm_type="layernorm",
        q_activation="silu",
        causal=True,
        use_embed_scale=False,
        # init
        init_type=0,
        token_mixer_init_type=0,
        rescale_type=0,
        channel_mixer_init_type=0,
        gain=0.02,
        ce_type="naive",
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
        self.token_mixer_norm_type = token_mixer_norm_type
        self.token_mixer_type = token_mixer_type
        # lrpe
        self.use_lrpe = use_lrpe
        self.lrpe_type = lrpe_type
        self.base = base
        # glu config
        self.mid_dim = mid_dim
        self.channel_mixer_type = channel_mixer_type
        self.channel_mixer_activation = channel_mixer_activation
        self.use_gate_linear = use_gate_linear
        # others
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.use_output_gate = use_output_gate
        self.norm_type = norm_type
        self.q_activation = q_activation
        self.causal = causal
        self.use_embed_scale = use_embed_scale
        self.chunk_type = chunk_type
        self.gradient_type = gradient_type
        self.use_init_weights = use_init_weights
        self.use_scale = use_scale
        self.chunk_size = chunk_size
        # init
        self.init_type = init_type
        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.channel_mixer_init_type = channel_mixer_init_type
        self.gain = gain
        self.ce_type = ce_type
