from .linear_attention import Hgru2, Hgru3, LinearAttention, MetaLa, TnlAttention
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
    # long conv
    "gtu": Gtu,
}


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
            head_dim=config.head_dim,
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
        )
