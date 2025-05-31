# GLU: https://arxiv.org/pdf/2002.05202.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from xmixers.modules.activations import get_activation_fn
from xmixers.utils import XMIXERS_DEBUG, print_module, print_params


class FastWeightGLU(nn.Module):
    def __init__(
        self,
        num_heads: int,
        fw_embed_dim: int,
        fw_mid_dim: int,
        fw_activation: str = "silu",
        bias: bool = False,
    ) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.w1 = nn.Parameter(torch.zeros(num_heads, fw_embed_dim, fw_mid_dim))
        self.w2 = nn.Parameter(torch.zeros(num_heads, fw_embed_dim, fw_mid_dim))
        self.w3 = nn.Parameter(torch.zeros(num_heads, fw_mid_dim, fw_embed_dim))
        self.act = get_activation_fn(fw_activation)

        self._init_weights()

    def extra_repr(self):
        return print_module(self)

    def _init_weights(self, gain=0.01):
        if getattr(self, "_is_hf_initialized", False):
            return

        nn.init.xavier_uniform_(self.w1, gain=gain)
        nn.init.xavier_uniform_(self.w2, gain=gain)
        nn.init.xavier_uniform_(self.w3, gain=gain)

        self._is_hf_initialized = True

    def init_fast_weight(self, b):
        with torch.enable_grad():
            self.w1.requires_grad = True
            self.w2.requires_grad = True
            self.w3.requires_grad = True

            return {
                "w1": repeat(self.w1, "h d e -> b h d e", b=b).contiguous(),
                "w2": repeat(self.w2, "h d e -> b h d e", b=b).contiguous(),
                "w3": repeat(self.w3, "h e d -> b h e d", b=b).contiguous(),
            }

    def forward(self, x, fast_weight):
        w1 = fast_weight["w1"]
        w2 = fast_weight["w2"]
        w3 = fast_weight["w3"]

        x1 = self.act(torch.einsum("b h d e, b n h d -> b n h e", w1, x))
        x2 = torch.einsum("b h d e, b n h d -> b n h e", w2, x)
        output = torch.einsum("b h e d, b n h e -> b n h d", w3, x1 * x2)

        return output


class FastWeightHpGLU(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_lr: bool = True,
        use_wd: bool = False,
        use_momentum: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.use_lr = use_lr
        self.use_wd = use_wd
        self.use_momentum = use_momentum

        if self.use_lr:
            self.w1_lr = nn.Linear(embed_dim, num_heads, bias=bias)
            self.w2_lr = nn.Linear(embed_dim, num_heads, bias=bias)
            self.w3_lr = nn.Linear(embed_dim, num_heads, bias=bias)

        if self.use_wd:
            self.w1_wd = nn.Linear(embed_dim, num_heads, bias=bias)
            self.w2_wd = nn.Linear(embed_dim, num_heads, bias=bias)
            self.w3_wd = nn.Linear(embed_dim, num_heads, bias=bias)

        if self.use_momentum:
            self.w1_momentum = nn.Linear(embed_dim, num_heads, bias=bias)
            self.w2_momentum = nn.Linear(embed_dim, num_heads, bias=bias)
            self.w3_momentum = nn.Linear(embed_dim, num_heads, bias=bias)

    def forward(self, x):
        lr_dict = {}
        if self.use_lr:
            lr_dict["w1"] = F.sigmoid(self.w1_lr(x))
            lr_dict["w2"] = F.sigmoid(self.w2_lr(x))
            lr_dict["w3"] = F.sigmoid(self.w3_lr(x))

        wd_dict = {}
        if self.use_wd:
            wd_dict["w1"] = F.sigmoid(self.w1_wd(x))
            wd_dict["w2"] = F.sigmoid(self.w2_wd(x))
            wd_dict["w3"] = F.sigmoid(self.w3_wd(x))

        momentum_dict = {}
        if self.use_momentum:
            momentum_dict["w1"] = F.sigmoid(self.w1_momentum(x))
            momentum_dict["w2"] = F.sigmoid(self.w2_momentum(x))
            momentum_dict["w3"] = F.sigmoid(self.w3_momentum(x))

        return {"lr_dict": lr_dict, "wd_dict": wd_dict, "momentum_dict": momentum_dict}
