# coding=utf-8
""" PolarRnn configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PolarRnnConfig(PretrainedConfig):
    model_type = "polar_rnn"
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
        token_mixer_type="polar_rnn",
        embed_dim=1024,
        num_heads=8,
        bias=False,
        ##### channel mixer config
        channel_mixer_type="glu",
        mid_dim=1024,
        channel_mixer_activation="silu",
        use_gate_linear=False,
        ##### others
        max_position_embeddings=1024,
        use_output_gate=True,
        norm_type="layernorm",
        q_activation="silu",
        k_activation="silu",
        v_activation="silu",
        use_gamma=True,
        gamma_activation="pos",
        use_decay=True,
        scalar_decay=True,
        qkv_norm_type=2,
        norm_q=False,
        norm_v=False,
        causal=True,
        num_layers=12,
        use_embed_scale=False,
        pad_embed_dim=True,
        ##### init
        init_type=0,
        token_mixer_init_type=0,
        rescale_type=0,
        channel_mixer_init_type=0,
        gain=0.02,
        fuse_norm_add=True,
        ce_type="xopes_flce",
        debug=0,
        use_l2_norm=False,
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
        self.token_mixer_type = token_mixer_type
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        # channel mixer config
        self.channel_mixer_type = channel_mixer_type
        self.mid_dim = mid_dim
        self.channel_mixer_activation = channel_mixer_activation
        self.use_gate_linear = use_gate_linear
        # others
        self.max_position_embeddings = max_position_embeddings
        self.use_output_gate = use_output_gate
        self.norm_type = norm_type
        self.q_activation = q_activation
        self.k_activation = k_activation
        self.v_activation = v_activation
        self.use_gamma = use_gamma
        self.gamma_activation = gamma_activation
        self.use_decay = use_decay
        self.scalar_decay = scalar_decay
        self.qkv_norm_type = qkv_norm_type
        self.norm_q = norm_q
        self.norm_v = norm_v
        self.causal = causal
        self.use_embed_scale = use_embed_scale
        self.num_layers = num_layers
        # init
        self.init_type = init_type
        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.channel_mixer_init_type = channel_mixer_init_type
        self.gain = gain
        self.fuse_norm_add = fuse_norm_add
        self.ce_type = ce_type
        self.debug = debug
        self.use_l2_norm = use_l2_norm
