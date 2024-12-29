# coding=utf-8
""" Naive Hybrid configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NaiveHybridConfig(PretrainedConfig):
    model_type = "naive_hybrid"
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
        token_mixer_type_list=[],
        causal=True,
        # attention config
        token_mixer_type="attn",
        embed_dim=1024,
        num_heads=8,
        kv_heads=-1,
        bias=False,
        softmax_use_lrpe=True,
        softmax_lrpe_type=1,
        softmax_base=10000,
        mpa_type=0,
        mpa_activation="none",
        head_dim=-1,
        kv_rank=2,
        cp_activation="none",
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        window_size=-1,
        # linear attention config
        use_output_gate=True,
        q_activation="silu",
        k_activation="silu",
        beta_activation="silu",
        use_dense_memory=True,
        n_min=2,
        n_max=256,
        linear_use_lrpe=False,
        linear_lrpe_type=1,
        linear_base=10000,
        # channel mixer config
        channel_mixer_type="glu",
        mid_dim=1024,
        channel_mixer_activation="silu",
        # for alu and lalu
        qk_dim=1024,
        v_dim=1024,
        mem_dim=1024,
        use_scale=0,
        output_gate_activation="silu",
        use_low_rank_output_gate=False,
        channel_mixer_init_type=0,
        # others
        max_position_embeddings=1024,
        num_layers=24,
        norm_type="layernorm",
        token_mixer_init_type=0,
        init_type=0,
        rescale_type=0,
        use_postnorm=False,
        use_embed_scale=False,
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
        self.token_mixer_type_list = token_mixer_type_list
        self.causal = causal
        # token mixer config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.bias = bias
        self.softmax_use_lrpe = softmax_use_lrpe
        self.softmax_lrpe_type = softmax_lrpe_type
        self.softmax_base = softmax_base
        self.token_mixer_type = token_mixer_type
        self.mpa_type = mpa_type
        self.mpa_activation = mpa_activation
        self.kv_rank = kv_rank
        self.cp_activation = cp_activation
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.head_dim = head_dim
        self.window_size = window_size
        # linear attention config
        self.use_output_gate = use_output_gate
        self.q_activation = q_activation
        self.k_activation = k_activation
        self.beta_activation = beta_activation
        self.use_dense_memory = use_dense_memory
        self.n_min = n_min
        self.n_max = n_max
        self.linear_use_lrpe = linear_use_lrpe
        self.linear_lrpe_type = linear_lrpe_type
        self.linear_base = linear_base
        # channel mixer config
        self.channel_mixer_type = channel_mixer_type
        self.mid_dim = mid_dim
        self.channel_mixer_activation = channel_mixer_activation
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.mem_dim = mem_dim
        self.use_scale = use_scale
        self.output_gate_activation = output_gate_activation
        self.use_low_rank_output_gate = use_low_rank_output_gate
        self.channel_mixer_init_type = channel_mixer_init_type
        # others
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.token_mixer_init_type = token_mixer_init_type
        self.init_type = init_type
        self.rescale_type = rescale_type
        self.use_postnorm = use_postnorm
        self.use_embed_scale = use_embed_scale
        self.ce_type = ce_type
