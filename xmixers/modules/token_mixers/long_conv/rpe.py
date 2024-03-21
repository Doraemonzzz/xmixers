import torch
import torch.nn as nn

from xmixers.utils import ActLayer, get_norm_fn, print_params


class Rpe(nn.Module):
    def __init__(
        self,
        in_dim,
        feature_dim,
        out_dim,
        activation="silu",
        bias=False,
        rpe_layers=3,
        norm_type="layernorm",
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.in_dim = in_dim
        if in_dim > 1:
            theta = 10000 ** (-2 / in_dim * torch.arange(in_dim // 2)).reshape(1, -1)
            self.register_buffer("theta", theta)

        self.pos_proj = nn.Linear(in_dim, feature_dim, bias=bias)
        self.layers = nn.ModuleList([])
        for _ in range(rpe_layers):
            self.layers.append(
                nn.Sequential(
                    get_norm_fn(norm_type)(feature_dim),
                    ActLayer(activation),
                    nn.Linear(feature_dim, feature_dim, bias=bias),
                )
            )
        self.out = nn.Sequential(
            get_norm_fn(norm_type)(feature_dim),
            ActLayer(activation),
            nn.Linear(feature_dim, out_dim, bias=bias),
        )

    def get_feature(self, index):
        if self.in_dim > 1:
            theta = index * self.theta
            x = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        else:
            x = index

        return x

    def forward(self, index):
        input = self.get_feature(index)
        x = self.pos_proj(input)
        for m in self.layers:
            x = m(x) + x
        x = self.out(x)

        return x
