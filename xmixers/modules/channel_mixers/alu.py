# Attention Linear Unit: https://arxiv.org/pdf/1706.03762v3

import torch
import torch.nn as nn
from einops import rearrange

from xmixers.modules.activations import get_activation_fn
from xmixers.utils import XMIXERS_DEBUG, print_module, print_params


class ALU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        qk_dim: int,
        v_dim: int,
        mem_dim: int,
        num_heads: int,
        activation: str,
        bias: bool = False,
        use_scale: int = 0,
        use_output_gate: bool = False,
        output_gate_activation: str = "silu",
        use_low_rank_output_gate: bool = False,
        channel_mixer_init_type: int = 0,
    ) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.q_proj = nn.Linear(embed_dim, qk_dim, bias=bias)
        self.k_weight = nn.Parameter(torch.randn(mem_dim, qk_dim), requires_grad=True)
        self.v_weight = nn.Parameter(torch.randn(mem_dim, v_dim), requires_grad=True)
        self.out_proj = nn.Linear(v_dim, embed_dim, bias=bias)
        self.use_output_gate = use_output_gate
        if self.use_output_gate:
            if use_low_rank_output_gate:
                mid_dim = embed_dim // num_heads
                self.output_gate = nn.Sequential(
                    nn.Linear(embed_dim, mid_dim, bias=bias),
                    nn.Linear(mid_dim, v_dim, bias=bias),
                )
            else:
                self.output_gate = nn.Linear(embed_dim, v_dim, bias=bias)
            self.output_gate_act = get_activation_fn(output_gate_activation)

        self.num_heads = num_heads
        self.use_scale = use_scale
        self.act = get_activation_fn(activation)
        self.channel_mixer_init_type = channel_mixer_init_type

        self._initialize_weights()

    def _initialize_weights(self):
        if self.channel_mixer_init_type == 0:
            nn.init.normal_(self.k_weight, mean=0.0, std=0.05)
            nn.init.normal_(self.v_weight, mean=0.0, std=0.05)
        elif self.channel_mixer_init_type == 1:  # fla init
            nn.init.xavier_uniform_(self.k_weight, gain=2**-2.5)
            nn.init.xavier_uniform_(self.v_weight, gain=2**-2.5)
        elif self.channel_mixer_init_type == 2:  # fairseq init
            nn.init.xavier_uniform_(self.k_weight, gain=2**-0.5)
            nn.init.xavier_uniform_(self.v_weight, gain=2**-0.5)
        elif self.channel_mixer_init_type == 3:
            nn.init.normal_(self.k_weight, mean=0.0, std=0.2)
            nn.init.normal_(self.v_weight, mean=0.0, std=0.2)

    def extra_repr(self):
        return print_module(self)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_weight
        v = self.v_weight
        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", h=self.num_heads),
            [q, k, v],
        )

        if self.use_scale == 1:
            scale = q.shape[-1] ** 0.5
        elif self.use_scale == 2:
            scale = x.shape[-1] ** 0.5
        else:
            scale = 1

        # v1
        # energy = self.act(torch.einsum("... n d, ... m d -> ... n m", q, k) * scale)

        # v2
        k = k * scale
        v = v * scale
        energy = self.act(torch.einsum("... n d, ... m d -> ... n m", q, k))
        output = torch.einsum("... n m, ... m d -> ... n d", energy, v)

        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")

        if self.use_output_gate:
            output_gate = self.output_gate_act(self.output_gate(x))
            output = output * output_gate

        # outproj
        output = self.out_proj(output)

        return output
