# coding=utf-8
# Tnl: https://arxiv.org/pdf/2405.17381
""" Tnl configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class TnlConfig(PretrainedConfig):
    model_type = "tnl"
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
        token_mixer_type="tnl_attn",
        embed_dim=1024,
        num_heads=8,
        bias=False,
        use_lrpe=True,
        lrpe_type=1,
        base=10000,
        use_output_gate=True,
        norm_type="rmsnorm",
        q_activation="silu",
        k_activation="silu",
        v_activation="silu",
        q_norm=False,
        k_norm=False,
        v_norm=False,
        causal=True,
        use_initial_state=False,
        gate_act="sigmoid",
        gate_pos="pre",
        token_mixer_norm_type="rmsnorm",
        ###### channel mixer config
        channel_mixer_type="glu",
        mid_dim=1024,
        channel_mixer_activation="silu",
        use_gate_linear=True,
        # others
        max_position_embeddings=1024,
        num_layers=24,
        use_lrpe_list=[False],
        n_min=2,
        n_max=256,
        use_embed_scale=False,
        ce_type="xopes_flce",
        # init
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
        ##### hf origin
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.init_std = init_std
        ##### add
        # token mixer config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        self.use_lrpe = use_lrpe
        self.lrpe_type = lrpe_type
        self.base = base
        self.use_output_gate = use_output_gate
        self.norm_type = norm_type
        self.q_activation = q_activation
        self.k_activation = k_activation
        self.v_activation = v_activation
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.v_norm = v_norm
        self.causal = causal
        self.use_initial_state = use_initial_state
        self.token_mixer_type = token_mixer_type
        self.gate_act = gate_act
        self.gate_pos = gate_pos
        self.token_mixer_norm_type = token_mixer_norm_type
        # channel mixer config
        self.channel_mixer_type = channel_mixer_type
        self.mid_dim = mid_dim
        self.channel_mixer_activation = channel_mixer_activation
        self.use_gate_linear = use_gate_linear
        # others
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.use_lrpe_list = use_lrpe_list
        self.n_min = n_min
        self.n_max = n_max
        self.use_embed_scale = use_embed_scale
        self.ce_type = ce_type
        self.token_mixer_init_type = token_mixer_init_type
        self.channel_mixer_init_type = channel_mixer_init_type
        self.init_type = init_type
        self.rescale_type = rescale_type
        self.gain = gain
