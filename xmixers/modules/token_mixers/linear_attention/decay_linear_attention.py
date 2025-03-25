import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from fla.ops.common.fused_recurrent import fused_recurrent
from fla.ops.gla import chunk_gla
from fla.ops.simple_gla import chunk_simple_gla
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache
from xopes.ops import cumsum_fn

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.modules.pes import Lrpe, get_log_slopes_general
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params


class DecayLinearAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_lrpe: bool = True,
        base: int = 10000,
        use_output_gate: bool = True,
        token_mixer_norm_type: str = "rmsnorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        decay_type: str = "hgrn2",  # choose from ["hgrn2", "gla", "mamba", "mamba_no_a_no_t", "mamba_no_a", "mamba_no_t", "lightnet", "tnl", "tnll", "lssp", "hgrn3"] # lssp: log sum soft plus
        # decay parameters
        A_init_range: tuple[float, float] = (1, 16),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        gate_denom: float = 16,
        threshold: float = 0.99,
        share_decay: bool = False,
        scalar_decay: bool = False,
        causal: bool = True,
        gate_act: str = "sigmoid",
        gate_pos: str = "pre",
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        assert causal, f"Only causal={causal} is supported"
        if decay_type in ["tnl", "tnll"]:
            assert scalar_decay, "When using tnl or tnll, scalar_decay must be True"

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.use_output_gate = use_output_gate

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.scalar_decay = scalar_decay
        self.share_decay = share_decay
        self.decay_type = decay_type
        if self.decay_type not in ["tnl", "tnll"]:
            if self.scalar_decay:
                self.f_proj = nn.Linear(embed_dim, self.num_heads, bias=bias)
                self.decay_dim = self.num_heads
            else:
                if not self.share_decay:
                    self.f_proj = nn.Sequential(
                        nn.Linear(embed_dim, self.head_dim, bias=bias),
                        nn.Linear(self.head_dim, embed_dim, bias=bias),
                    )
                self.decay_dim = embed_dim

        self.gate_denom = gate_denom

        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        norm_type = (
            f"{token_mixer_norm_type}_fused_gate"
            if use_output_gate
            else token_mixer_norm_type
        )
        self.norm = get_norm_fn(norm_type)(
            embed_dim,
            bias=bias,
            gate_act=gate_act,
            gate_pos=gate_pos,
            num_groups=num_heads,
        )

        self.use_lrpe = use_lrpe
        if self.use_lrpe:
            self.lrpe = Lrpe(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                lrpe_type=6,
                base=base,
            )

        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )

        self.threshold = threshold
        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self.decay_state = torch.empty(0)

        self.decay_kwargs = {
            "A_init_range": A_init_range,
            "dt_min": dt_min,
            "dt_max": dt_max,
            "dt_init_floor": dt_init_floor,
            "dt_limit": dt_limit,
        }
        self.setup_decay(**self.decay_kwargs)
        self._init_weights()

    def _init_weights(self):
        self.setup_decay(**self.decay_kwargs)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        self.setup_decay(**self.decay_kwargs)
        return _initialize_weights(self, module)

    def extra_repr(self):
        return print_module(self)

    def save_decay(self, log_f, **kwargs):
        save_dir = os.path.join(
            kwargs["save_dir"],
        )
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        data = log_f.float().cpu().numpy()
        file_path = os.path.join(save_dir, f"layer_{self.layer_idx}.npy")
        np.save(file_path, data)

    def setup_decay(self, **kwargs):
        if self.decay_type == "hgrn2":
            pass
        elif self.decay_type == "gla":
            pass
        elif self.decay_type in [
            "mamba",
            "mamba_no_a_no_t",
            "mamba_no_a",
            "mamba_no_t",
        ]:
            A_init_range = kwargs["A_init_range"]
            dt_max = kwargs["dt_max"]
            dt_min = kwargs["dt_min"]
            dt_init_floor = kwargs["dt_init_floor"]
            assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]

            if self.decay_type in ["mamba", "mamba_no_t"]:
                A = torch.empty(self.decay_dim, dtype=torch.float32).uniform_(
                    *A_init_range
                )
                log_A = torch.log(A)
                if hasattr(self, "log_A"):
                    if isinstance(self.log_A, DTensor):
                        self.log_A.data.copy_(
                            DTensor.from_local(
                                log_A, device_mesh=self.log_A.device_mesh
                            )
                        )
                    else:
                        self.log_A.data.copy_(log_A)
                else:
                    self.log_A = nn.Parameter(log_A, requires_grad=True)

            if self.decay_type in [
                "mamba",
                "mamba_no_a",
            ]:
                dt = torch.exp(
                    torch.rand(
                        self.decay_dim,
                    )
                    * (math.log(dt_max) - math.log(dt_min))
                    + math.log(dt_min)
                )
                dt = torch.clamp(dt, min=dt_init_floor)
                # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                inv_dt = dt + torch.log(-torch.expm1(-dt))  # expm1(x) = exp(x) - 1

                if hasattr(self, "dt_bias"):
                    if isinstance(self.dt_bias, DTensor):
                        self.dt_bias.data.copy_(
                            DTensor.from_local(
                                inv_dt, device_mesh=self.dt_bias.device_mesh
                            )
                        )
                    else:
                        self.dt_bias.data.copy_(inv_dt.to(self.dt_bias.dtype))
                else:
                    self.dt_bias = nn.Parameter(inv_dt, requires_grad=True)
        elif self.decay_type == "lightnet":
            pass
        elif self.decay_type == "tnl":
            self.log_decay = torch.empty(0)
        elif self.decay_type == "tnll":
            log_decay = get_log_slopes_general(self.num_heads) * (
                1 - self.layer_idx / (self.num_layers - 1) + 1e-5
            )

            if hasattr(self, "log_decay"):
                if isinstance(self.log_decay, DTensor):
                    self.log_decay.data.copy_(
                        DTensor.from_local(
                            log_decay, device_mesh=self.log_decay.device_mesh
                        )
                    )
                else:
                    self.log_decay.data.copy_(log_decay)
            else:
                self.log_decay = nn.Parameter(log_decay, requires_grad=True)
        elif self.decay_type == "lssp":
            pass
        elif self.decay_type == "hgrn3":
            # take x = 0 as median, 1 / (1 + exp(-(median + delta))) = a => 1 + exp(-delta) = 1 / a => exp(-delta) = (1 / a - 1) -> exp(delta) = a / (1 - a) => delta = log(a / (1 - a))
            a = self.threshold
            delta = torch.ones(self.decay_dim) * math.log(a / (1 - a))
            if hasattr(self, "delta"):
                if isinstance(self.delta, DTensor):
                    self.delta.data.copy_(
                        DTensor.from_local(
                            delta,
                            device_mesh=self.delta.device_mesh,
                        )
                    )
                else:
                    self.delta.data.copy_(delta)
            else:
                self.delta = nn.Parameter(delta, requires_grad=True)
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

    def compute_decay(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        lower_bound: Optional[torch.Tensor] = None,
        q_offset: int = 0,
        decay_state: Optional[torch.Tensor] = None,
        use_attn_mask: bool = False,
        **kwargs,
    ):
        if self.decay_type == "hgrn2":
            if self.share_decay:
                f = F.sigmoid(self.k_proj(x))
            else:
                k = self.k_act(self.k_proj(x))
                f = F.sigmoid(self.f_proj(x))
            # l + (1 - l) * sigmoid(x)
            if lower_bound is not None:
                f = lower_bound + (1 - lower_bound) * f
                if self.share_decay:
                    k = 1 - f
            log_f = torch.log(f)
            decay_state = None
        elif self.decay_type == "gla":
            if self.share_decay:
                f = F.sigmoid(self.k_proj(x) / self.gate_denom)
                k = 1 - f
            else:
                k = self.k_act(self.k_proj(x))
                f = F.sigmoid(self.f_proj(x) / self.gate_denom)

            log_f = torch.log(f)
            decay_state = None
        elif self.decay_type == "mamba":
            if self.share_decay:
                k = F.softplus(self.k_proj(x) + self.dt_bias)
                log_f = -self.log_A.float().exp() * k
            else:
                k = self.k_act(self.k_proj(x))
                log_f = -self.log_A.float().exp() * F.softplus(
                    self.f_proj(x) + self.dt_bias
                )
            decay_state = None
        elif self.decay_type in [
            "mamba_no_a",
        ]:
            if self.share_decay:
                k = F.softplus(self.f_proj(x) + self.dt_bias)
                log_f = -k
            else:
                k = self.k_act(self.k_proj(x))
                log_f = -F.softplus(self.f_proj(x) + self.dt_bias)
            decay_state = None
        elif self.decay_type in [
            "mamba_no_t",
        ]:
            if self.share_decay:
                k = F.softplus(self.f_proj(x))
                log_f = -self.log_A.float().exp() * k
            else:
                k = self.k_act(self.k_proj(x))
                log_f = -self.log_A.float().exp() * F.softplus(self.f_proj(x))
            decay_state = None
        elif self.decay_type in [
            "mamba_no_a_no_t",
        ]:
            if self.share_decay:
                k = F.softplus(self.f_proj(x))
                log_f = -k
            else:
                k = self.k_act(self.k_proj(x))
                log_f = -F.softplus(self.f_proj(x))
            decay_state = None
        elif self.decay_type in ["lightnet", "lssp"]:
            b, n, d = x.shape

            shape = (b, 1, self.decay_dim)
            value = -float("inf")

            if self.decay_state.shape[0] == 0 or self.decay_state.shape[0] != b:
                self.decay_state = torch.zeros(
                    shape, device=x.device, dtype=torch.float32
                )
            if decay_state is None:
                decay_state = self.decay_state

            if self.share_decay:
                f = self.k_proj(x)
            else:
                f = self.f_proj(x)
                k = self.k_proj(x)

            if use_attn_mask:
                start = q_offset
                attention_mask_ = attention_mask[:, start:].unsqueeze(-1)
                f = f.masked_fill(attention_mask_ == 0, value)

            f = torch.cat([decay_state, f], dim=1)
            if self.decay_type == "lightnet":
                z = f.float().logcumsumexp(1)
            else:
                z = torch.log(cumsum_fn(F.softplus(f), dim=1))

            log_f = (z[:, :-1] - z[:, 1:]).to(f.dtype)
            decay_state = z[:, -1:]

            if self.share_decay:
                k = torch.exp(f[:, :-1] - z[:, 1:])
            else:
                k = self.k_act(k)
        elif self.decay_type in ["tnl", "tnll"]:
            if self.log_decay.shape[0] == 0:
                self.log_decay = (
                    get_log_slopes_general(self.num_heads)
                    * (1 - self.layer_idx / (self.num_layers - 1) + 1e-5)
                ).to(x.device)
            b, n, d = x.shape
            k = self.k_act(self.k_proj(x))
            log_f = repeat(self.log_decay, "h -> b n h", b=b, n=n)
            decay_state = None
        elif self.decay_type == "hgrn3":
            if self.share_decay:
                log_f = F.logsigmoid(self.k_proj(x) + self.delta)
                k = 1 - torch.exp(log_f)
            else:
                k = self.k_act(self.k_proj(x))
                log_f = F.logsigmoid(self.f_proj(x) + self.delta)
            decay_state = None
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

        if kwargs.get("save_decay", False):
            self.save_decay(log_f, **kwargs)

        return k, log_f, decay_state

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        b, n, d = x.shape
        # linear map
        q = self.q_proj(x)
        v = self.v_proj(x)

        # act
        q = self.q_act(q)

        q_offset = 0
        recurrent_state = None
        decay_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"][0]
            decay_state = past_key_values[self.layer_idx]["recurrent_state"][1]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )
        k, log_f, decay_state = self.compute_decay(
            x=x,
            attention_mask=attention_mask,
            lower_bound=lower_bound,
            q_offset=q_offset,
            decay_state=decay_state,
            use_attn_mask=use_attn_mask,
            **kwargs,
        )

        # left padding
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1)
            log_f = log_f.masked_fill(attention_mask_ == 0, 0)
            k = k.masked_fill(attention_mask_ == 0, 0)

        q, k, v = map(
            lambda x: rearrange(
                x, "b n (h d) -> b n h d", d=self.head_dim
            ).contiguous(),
            [q, k, v],
        )
        vector_decay = self.decay_type not in ["tnl", "tnll"] and (
            not self.scalar_decay
        )
        if vector_decay:
            log_f = rearrange(log_f, "... (h d) -> ... h d", d=self.head_dim)

        if self.use_lrpe:
            q = self.lrpe(q, offset=q_offset)
            k = self.lrpe(k, offset=q_offset)
            if (
                vector_decay
            ):  # we only use complex lrpe when vector_decay is True, so we need to repeat log_f
                log_f = repeat(log_f, "... d -> ... (g d)", g=2)

        dtype = q.dtype
        q, k, v, log_f = map(lambda x: x.to(dtype), [q, k, v, log_f])

        scale = 1
        if self.causal:
            if self.training or recurrent_state is None:  # training or prefilling
                if self.scalar_decay:
                    fn = chunk_simple_gla
                else:
                    fn = chunk_gla
                output, recurrent_state = fn(
                    q=q,
                    k=k,
                    v=v,
                    g=log_f,
                    scale=scale,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    head_first=False,
                )
            else:
                if self.scalar_decay:
                    g = log_f
                    gk = None
                else:
                    g = None
                    gk = log_f

                output, recurrent_state = fused_recurrent(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    gk=gk,
                    scale=scale,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    head_first=False,
                )
        else:
            assert False

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=[recurrent_state, decay_state],
                layer_idx=self.layer_idx,
                offset=n,
            )

        # reshape
        output = rearrange(output, "... n h d -> ... n (h d)")
        if self.use_output_gate:
            gate = self.output_gate(x)
            output = self.norm(output, gate)
        else:
            output = self.norm(output)

        # out proj
        output = self.o_proj(output)

        return output, past_key_values
