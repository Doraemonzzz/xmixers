from .linear_attention import (
    Hgru2,
    Hgru3,
    LinearAttention,
    MetaLa,
    PolarRnn,
    TnlAttention,
)
from .long_conv import Gtu
from .vanilla_attention import (
    Attention,
    CPAttention,
    FlexAttention,
    MultiLatentAttention,
    MultiProductAttention,
    nAttention,
)

AUTO_TOKEN_MIXER_MAPPING = {
    # softmax attn
    "attn": Attention,
    "flex_attn": FlexAttention,
    "n_attn": nAttention,
    "mpa": MultiProductAttention,
    "cpa": CPAttention,
    "mla": MultiLatentAttention,
    # linear attn
    "hgru2": Hgru2,
    "hgru3": Hgru3,
    "linear_attn": LinearAttention,
    "tnl_attn": TnlAttention,
    "metala": MetaLa,
    "polar_rnn": PolarRnn,
    # long conv
    "gtu": Gtu,
}

SOFTMAX_TOKEN_MIXER_LIST = ["attn", "flex_attn", "n_attn", "mpa", "cpa", "mla"]
LINEAR_TOKEN_MIXER_LIST = [
    "hgru2",
    "hgru3",
    "linear_attn",
    "tnl_attn",
    "metala",
    "polar_rnn",
]


def get_token_mixer(config, layer_idx):
    cls = AUTO_TOKEN_MIXER_MAPPING[config.token_mixer_type]
    if config.token_mixer_type in ["attn", "flex_attn", "n_attn"]:
        return cls(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            kv_heads=config.kv_heads,
            bias=config.bias,
            use_lrpe=config.use_lrpe,
            layer_idx=layer_idx,
            lrpe_type=config.lrpe_type,
            base=config.base,
            max_position_embeddings=config.max_position_embeddings,
            token_mixer_init_type=config.token_mixer_init_type,
            rescale_type=config.rescale_type,
            num_layers=config.num_layers,
            window_size=config.window_size,
            init_std=config.init_std,
            gain=config.gain,
        )
    elif config.token_mixer_type in [
        "mpa",
    ]:
        return cls(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            kv_heads=config.kv_heads,
            bias=config.bias,
            use_lrpe=config.use_lrpe,
            layer_idx=layer_idx,
            lrpe_type=config.lrpe_type,
            base=config.base,
            max_position_embeddings=config.max_position_embeddings,
            token_mixer_init_type=config.token_mixer_init_type,
            rescale_type=config.rescale_type,
            num_layers=config.num_layers,
            mpa_type=config.mpa_type,
            mpa_activation=config.mpa_activation,
            gate_type=config.gate_type,
            head_dim=config.head_dim,
            init_std=config.init_std,
            gain=config.gain,
            use_l2_norm=config.use_l2_norm,
        )
    elif config.token_mixer_type in ["cpa"]:
        return cls(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            kv_heads=config.kv_heads,
            bias=config.bias,
            use_lrpe=config.use_lrpe,
            layer_idx=layer_idx,
            lrpe_type=config.lrpe_type,
            base=config.base,
            max_position_embeddings=config.max_position_embeddings,
            kv_rank=config.kv_rank,
            token_mixer_init_type=config.token_mixer_init_type,
            rescale_type=config.rescale_type,
            num_layers=config.num_layers,
            cp_activation=config.cp_activation,
            init_std=config.init_std,
        )
    elif config.token_mixer_type in ["mla"]:
        return cls(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            bias=config.bias,
            use_lrpe=config.use_lrpe,
            layer_idx=layer_idx,
            lrpe_type=config.lrpe_type,
            base=config.base,
            max_position_embeddings=config.max_position_embeddings,
            token_mixer_init_type=config.token_mixer_init_type,
            rescale_type=config.rescale_type,
            num_layers=config.num_layers,
            init_std=config.init_std,
        )
    elif config.token_mixer_type in ["hgru2"]:
        return cls(
            embed_dim=config.embed_dim,
            expand_ratio=config.expand_ratio,
            bias=config.bias,
            layer_idx=layer_idx,
            use_output_gate=config.use_output_gate,
            norm_type=config.norm_type,
            q_activation=config.q_activation,
            causal=config.causal,
            rescale_type=config.rescale_type,
            token_mixer_init_type=config.token_mixer_init_type,
            num_layers=config.num_layers,
            init_std=config.init_std,
            gain=config.gain,
            beta_activation=config.beta_activation,
            use_dense_memory=config.use_dense_memory,
            token_mixer_norm_type=config.token_mixer_norm_type,
            norm_pos=config.norm_pos,
        )
    elif config.token_mixer_type in ["hgru3"]:
        return cls(
            embed_dim=config.embed_dim,
            expand_ratio=config.expand_ratio,
            bias=config.bias,
            layer_idx=layer_idx,
            use_output_gate=config.use_output_gate,
            norm_type=config.norm_type,
            q_activation=config.q_activation,
            k_activation=config.k_activation,
            beta_activation=config.beta_activation,
            causal=config.causal,
            use_dense_memory=config.use_dense_memory,
            rescale_type=config.rescale_type,
            token_mixer_init_type=config.token_mixer_init_type,
            num_layers=config.num_layers,
            init_std=config.init_std,
        )
    elif config.token_mixer_type in ["linear_attn"]:
        return cls(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            kv_heads=config.kv_heads,
            bias=config.bias,
            use_lrpe=config.use_lrpe,
            layer_idx=layer_idx,
            lrpe_type=config.lrpe_type,
            base=config.base,
            use_output_gate=config.use_output_gate,
            norm_type=config.norm_type,
            linear_activation=config.linear_activation,
            causal=config.causal,
            use_dense_memory=config.use_dense_memory,
            max_position_embeddings=config.max_position_embeddings,
            token_mixer_init_type=config.token_mixer_init_type,
            rescale_type=config.rescale_type,
            num_layers=config.num_layers,
            init_std=config.init_std,
            gain=config.gain,
        )
    elif config.token_mixer_type in ["tnl_attn"]:
        return cls(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            kv_heads=config.kv_heads,
            bias=config.bias,
            use_lrpe=config.use_lrpe_list[layer_idx]
            if len(config.use_lrpe_list) > layer_idx
            else config.use_lrpe_list[0],
            layer_idx=layer_idx,
            lrpe_type=config.lrpe_type,
            base=config.base,
            gate_dim=config.gate_dim,
            use_output_gate=config.use_output_gate,
            norm_type=config.norm_type,
            q_activation=config.q_activation,
            k_activation=config.k_activation,
            v_activation=config.v_activation,
            causal=config.causal,
            norm_pos=config.norm_pos,
            max_position_embeddings=config.max_position_embeddings,
            token_mixer_init_type=config.token_mixer_init_type,
            rescale_type=config.rescale_type,
            num_layers=config.num_layers,
            init_std=config.init_std,
        )
    elif config.token_mixer_type in ["metala"]:
        return cls(
            embed_dim=config.embed_dim,
            expand_ratio=config.expand_ratio,
            bias=config.bias,
            layer_idx=layer_idx,
            use_output_gate=config.use_output_gate,
            non_sparse_ratio=config.non_sparse_ratio,
            num_sparse=config.num_sparse,
            norm_type=config.norm_type,
            q_activation=config.q_activation,
            causal=config.causal,
            rescale_type=config.rescale_type,
            token_mixer_init_type=config.token_mixer_init_type,
            num_layers=config.num_layers,
            init_std=config.init_std,
        )
    elif config.token_mixer_type in ["polar_rnn"]:
        return cls(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            bias=config.bias,
            layer_idx=layer_idx,
            use_output_gate=config.use_output_gate,
            norm_type=config.norm_type,
            q_activation=config.q_activation,
            k_activation=config.k_activation,
            v_activation=config.v_activation,
            use_gamma=config.use_gamma,
            gamma_activation=config.gamma_activation,
            use_decay=config.use_decay,
            scaler_decay=config.scaler_decay,
            qkv_norm_type=config.qkv_norm_type,
            norm_q=config.norm_q,
            norm_v=config.norm_v,
            causal=config.causal,
            token_mixer_init_type=config.token_mixer_init_type,
            rescale_type=config.rescale_type,
            num_layers=config.num_layers,
            init_std=config.init_std,
            gain=config.gain,
            debug=config.debug,
        )
