# adapt from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2_simple.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.cache_utils import Cache
from xopes.ops import lightning_attn_func

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_update = None
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    selective_state_update = None
    mamba_chunk_scan_combined = None


from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params


class Mamba2(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        conv_init: Optional[float] = None,
        expand: int = 2,
        headdim: int = 128,
        ngroups: int = 1,
        A_init_range: Tuple[float, float] = (1, 16),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        activation: str = "swish",
        bias: bool = False,
        conv_bias: bool = True,
        # Fused kernel and sharding options
        chunk_size: int = 256,
        layer_idx: int = 0,
        token_mixer_norm_type: str = "rmsnorm",
        gate_act: str = "sigmoid",
        gate_pos: str = "pre",
        causal: bool = True,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.01,
        use_lightning: bool = False,
        **kwargs,
    ):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        d_model = embed_dim
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.dt_init_floor = dt_init_floor
        self.A_init_range = A_init_range

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.o_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.setup_mamba_params()

        # Extra normalization layer right before output projection
        num_groups = self.d_inner // self.headdim
        norm_type = f"{token_mixer_norm_type}_fused_gate"
        self.norm = get_norm_fn(norm_type)(
            self.d_inner,
            bias=bias,
            gate_act=gate_act,
            gate_pos=gate_pos,
            num_groups=num_groups,
        )

        if use_lightning:
            self.forward = self.forward_simple

        self.causal = causal
        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self._init_weights()

    def extra_repr(self):
        return print_module(self)

    def setup_mamba_params(self):
        # Initialize conv1d
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        )
        dt = torch.clamp(dt, min=self.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        if hasattr(self, "dt_bias"):
            if isinstance(self.dt_bias, DTensor):
                self.dt_bias.data.copy_(
                    DTensor.from_local(
                        inv_dt,
                        device_mesh=self.dt_bias.device_mesh,
                    )
                )
            else:
                self.dt_bias.data.copy_(inv_dt)
        else:
            self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert self.A_init_range[0] > 0 and self.A_init_range[1] >= self.A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*self.A_init_range)
        A_log = torch.log(A)
        if hasattr(self, "A_log"):
            if isinstance(self.A_log, DTensor):
                self.A_log.data.copy_(
                    DTensor.from_local(
                        A_log,
                        device_mesh=self.A_log.device_mesh,
                    )
                )
            else:
                self.A_log.data.copy_(A_log)
        else:
            self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        D = torch.ones(self.nheads)
        if hasattr(self, "D"):
            if isinstance(self.D, DTensor):
                self.D.data.copy_(
                    DTensor.from_local(
                        D,
                        device_mesh=self.D.device_mesh,
                    )
                )
            else:
                self.D.data.copy_(D)
        else:
            self.D = nn.Parameter(D)
        self.D._no_weight_decay = True

    def _init_weights(self):
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        return _initialize_weights(self, module)

    def forward(
        self,
        u,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        recurrent_state = None
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        assert self.activation in ["silu", "swish"]

        q_offset = 0
        conv_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            conv_state = past_key_values[self.layer_idx]["conv_state"][0]
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"][0]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (seqlen > 1)
        )
        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1)
            xBC = xBC.masked_fill(attention_mask_ == 0, 0)
            dt = dt.masked_fill(attention_mask_ == 0, 0)

        # 1D Convolution
        if self.training or conv_state is None:  # training or prefilling
            if not self.training and conv_state is None:  # !!! important, update first
                w = self.d_conv
                l = min(w, seqlen)
                # b d w
                conv_state = xBC.new_zeros(batch, xBC.shape[-1], w)
                conv_state[:, :, -l:] = xBC[:, -l:, :].transpose(1, 2)

            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        else:
            xBC = causal_conv1d_update(
                x=xBC.transpose(1, 2),
                conv_state=conv_state,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        xBC = xBC.transpose(1, 2)

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        if (
            self.training or recurrent_state is None or seqlen > 1
        ):  # training or prefilling
            y, recurrent_state = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                initial_states=recurrent_state,
                return_final_states=True,
                **dt_limit_kwargs,
            )
        else:
            dtype = u.dtype
            dt = dt.squeeze(1)
            x = x.squeeze(1)
            B = B.squeeze(1)
            C = C.squeeze(1)
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            recurrent_state.copy_(
                recurrent_state * rearrange(dA, "b h -> b h 1 1") + dBx
            )
            y = torch.einsum("bhpn,bn->bhp", recurrent_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = y.unsqueeze(1)

        if past_key_values is not None:
            past_key_values.update(
                conv_state=[conv_state],
                recurrent_state=[recurrent_state],
                layer_idx=self.layer_idx,
                offset=seqlen,
            )

        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.o_proj(y)

        return out, past_key_values

    def forward_simple(
        self,
        u,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        b, n, d = u.shape

        # b n d_in_proj
        zxbcdt = self.in_proj(u)
        # h
        A = -torch.exp(self.A_log)
        recurrent_state = None

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )

        q_offset = 0
        conv_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            conv_state = past_key_values[self.layer_idx]["conv_state"][0]
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"][0]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )
        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1)
            xBC = xBC.masked_fill(attention_mask_ == 0, 0)
            dt = dt.masked_fill(attention_mask_ == 0, 0)

        # 1D Convolution
        if self.training or conv_state is None:  # training or prefilling
            if not self.training and conv_state is None:  # !!! important, update first
                w = self.d_conv
                l = min(w, n)
                # b d w
                conv_state = xBC.new_zeros(b, xBC.shape[-1], w)
                conv_state[:, :, -l:] = xBC[:, -l:, :].transpose(1, 2)

            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        else:
            xBC = causal_conv1d_update(
                x=xBC.transpose(1, 2),
                conv_state=conv_state,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        xBC = xBC.transpose(1, 2)

        # x: v, B: k, C: q
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        q = rearrange(C, "b n (g d) -> b n g d", d=self.d_state)
        k = rearrange(B, "b n (g d) -> b n g d", d=self.d_state)
        v = rearrange(x, "b n (h d) -> b n h d", d=self.headdim)
        g = q.shape[-2]
        h = v.shape[-2]

        # b n h d
        q, k = map(lambda x: repeat(x, "b n g d -> b n (g h) d", h=h // g), (q, k))
        # b n h
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
        log_f = dt * A
        k = dt.unsqueeze(-1) * k

        if self.causal:
            output, recurrent_state = lightning_attn_func(
                q=q,
                k=k,
                v=v,
                ld=log_f,
                initial_state=recurrent_state,
                decay_type="scalar",
            )
        else:
            assert False

        if past_key_values is not None:
            past_key_values.update(
                conv_state=[conv_state],
                recurrent_state=[recurrent_state],
                layer_idx=self.layer_idx,
                offset=n,
            )

        output = output + self.D.to(v.dtype).unsqueeze(-1) * v

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        output = self.norm(output, z)

        # out proj
        output = self.o_proj(output)

        return output, past_key_values
