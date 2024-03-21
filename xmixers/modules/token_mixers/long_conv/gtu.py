import torch.nn as nn

from xmixers.utils import get_activation_fn

from .tno import Tno


class Gtu(nn.Module):
    def __init__(
        self,
        embed_dim,
        expand_ratio=1,
        bias=False,
        activation="silu",
        causal=False,
        norm_type="layernorm",
        use_decay=True,
        in_dim=1,
        feature_dim=32,
        rpe_layers=3,
        dims=[1],
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio

        d1 = int(self.expand_ratio * embed_dim)
        # linear projection
        self.uv_proj = nn.Linear(embed_dim, 2 * d1, bias=bias)
        self.o = nn.Linear(d1, embed_dim, bias=bias)
        self.act = get_activation_fn(activation)
        self.dims = dims
        self.tno_list = nn.ModuleList([])
        for dim in dims:
            # tno
            self.tno_list.append(
                Tno(
                    in_dim=in_dim,
                    feature_dim=feature_dim,
                    out_dim=d1,
                    activation=activation,
                    bias=bias,
                    rpe_layers=rpe_layers,
                    norm_type=norm_type,
                    use_decay=use_decay,
                    causal=causal,
                    dim=dim,
                )
            )

    def forward(self, x):
        # x: b, n, d
        u, v = self.act(self.uv_proj(x)).chunk(2, dim=-1)
        output = 0
        for tno in self.tno_list:
            output += tno(v)
        output = u * output

        output = self.o(output)

        return output
