# Linear Attention Linear Unit

import torch.nn as nn

from xmixers.modules.activations import get_activation_fn
from xmixers.utils import XMIXERS_DEBUG, print_module, print_params


class LALU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        qk_dim: int,
        v_dim: int,
        mem_dim: int,
        num_heads: int,
        activation: str,
        bias: bool = False,
        use_scale=False,
        use_output_gate: bool = False,
        output_gate_activation: str = "silu",
        channel_mixer_init_type=0,
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
            self.out_gate = nn.Linear(embed_dim, v_dim, bias=bias)
            self.output_gate_act = get_activation_fn(output_gate_activation)

        self.num_heads = num_heads
        self.act = get_activation_fn(activation)
        self.channel_mixer_init_type = channel_mixer_init_type

        self._initialize_weights()

    def _initialize_weights(self):
        if self.channel_mixer_init_type == 0:
            pass
        elif self.channel_mixer_init_type == 1:  # fla init
            nn.init.xavier_uniform_(self.k_weight, gain=2**-2.5)
            nn.init.xavier_uniform_(self.v_weight, gain=2**-2.5)
        elif self.channel_mixer_init_type == 2:  # fairseq init
            nn.init.xavier_uniform_(self.k_weight, gain=2**-0.5)
            nn.init.xavier_uniform_(self.v_weight, gain=2**-0.5)

    def extra_repr(self):
        return print_module(self)

    def forward(self, x):
        q = self.act(self.q_proj(x))
        k = self.act(self.k_weight)
        v = self.v_weight
        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", h=self.num_heads),
            [q, k, v],
        )

        kv = torch.einsum("... n d, ... n e -> ... d e", k, v)
        output = torch.einsum("... n d, ... d e -> ... n e", q, kv)

        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")

        if self.use_output_gate:
            output_gate = self.output_gate_act(self.out_gate(x))
            output = output * output_gate

        # outproj
        output = self.out_proj(output)

        return output
