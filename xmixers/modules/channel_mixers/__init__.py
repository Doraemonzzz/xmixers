from .alu import ALU
from .ffn import FFN
from .glu import GLU
from .lalu import LALU
from .nglu import nGLU

AUTO_CHANNEL_MIXER_MAPPING = {
    "ffn": FFN,
    "glu": GLU,
    "nglu": nGLU,
    "alu": ALU,
    "lalu": LALU,
}


def get_channel_mixer(config):
    cls = AUTO_CHANNEL_MIXER_MAPPING[config.channel_mixer_type]
    if config.channel_mixer_type in ["ffn", "glu", "nglu"]:
        return cls(
            embed_dim=config.embed_dim,
            mid_dim=config.mid_dim,
            activation=config.channel_mixer_activation,
            bias=config.bias,
        )
    elif config.channel_mixer_type in ["alu", "lalu"]:
        return cls(
            embed_dim=config.embed_dim,
            qk_dim=config.qk_dim,
            v_dim=config.v_dim,
            mem_dim=config.mem_dim,
            num_heads=config.num_heads,
            activation=config.channel_mixer_activation,
            bias=config.bias,
            use_scale=config.use_scale,
            use_output_gate=config.use_output_gate,
            output_gate_activation=config.output_gate_activation,
            use_low_rank_output_gate=config.use_low_rank_output_gate,
            channel_mixer_init_type=config.channel_mixer_init_type,
        )
