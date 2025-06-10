# Hgru3: Hgru2 with negative decay
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.mesa_net import chunk_mesa_net, mesa_net_decoding_one_step
from torch.distributed.tensor import DTensor
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn, l2_norm
from xmixers.utils import XMIXERS_DEBUG, _initialize_weights, print_module, print_params


class MesaUnit(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        token_mixer_norm_type: str = "rmsnorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        threshold: float = 0.99,
        causal: bool = True,
        token_mixer_init_type: int = 4,
        rescale_type: int = 2,
        num_layers: int = 12,
        init_std: float = 0.02,
        gain: float = 0.01,
        gate_act: str = "sigmoid",
        gate_pos: str = "pre",
        lambda_initial_value: float = 1.0,
        lambda_lower_bound: float = 0.25,
        max_cg_step_training: int = 30,
        max_cg_step_decoding: int = 30,
        **kwargs,
    ):
        super().__init__()
        if XMIXERS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        assert causal, f"Only causal={causal} is supported"

        self.layer_idx = layer_idx
        self.causal = causal
        self.use_output_gate = use_output_gate
        self.max_cg_step_training = max_cg_step_training
        self.max_cg_step_decoding = max_cg_step_decoding
        self.decay_dim = num_heads
        self.lambda_initial_value = lambda_initial_value
        self.lambda_lower_bound = lambda_lower_bound
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.f_proj = nn.Linear(embed_dim, self.decay_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.bet_proj = nn.Linear(embed_dim, num_heads, bias=bias)

        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.threshold = threshold

        num_groups = num_heads
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
            num_groups=num_groups,
        )

        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, self.head_dim, bias=bias),
                nn.Linear(self.head_dim, embed_dim, bias=bias),
            )

        self.token_mixer_init_type = token_mixer_init_type
        self.rescale_type = rescale_type
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.gain = gain
        self._init_weights()

    def extra_repr(self):
        return print_module(self)

    def setup_decay(self):
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

        init_lamb_value = torch.log(
            torch.exp(
                torch.exp(
                    torch.tensor(self.lambda_initial_value - self.lambda_lower_bound)
                )
            )
            - 1.0
        )
        init_lamb_params = torch.empty(self.embed_dim, dtype=torch.float32).fill_(
            init_lamb_value
        )
        if hasattr(self, "lamb_params"):
            if isinstance(self.lamb_params, DTensor):
                self.lamb_params.data.copy_(
                    DTensor.from_local(
                        init_lamb_params,
                        device_mesh=self.lamb_params.device_mesh,
                    )
                )
            else:
                self.lamb_params.data.copy_(init_lamb_params)
        else:
            self.lamb_params = nn.Parameter(init_lamb_params, requires_grad=True)

    def _init_weights(self):
        self.setup_decay()
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        self.setup_decay()
        return _initialize_weights(self, module)

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        b, n, d = x.shape
        # x: b n d
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        log_f = F.logsigmoid(self.f_proj(x) + self.delta)
        lamb = F.softplus(self.lamb_params.float()) + self.lambda_lower_bound
        beta = F.sigmoid(self.bet_proj(x))

        # h is num_head, d is head dimension
        q, k, v, lamb = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim),
            [q, k, v, lamb],
        )

        # act
        q = l2_norm(q)
        k = l2_norm(k)

        q_offset = 0
        recurrent_state_kk = None
        recurrent_state_kv = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state_kk, recurrent_state_kv = past_key_values[self.layer_idx][
                "recurrent_state"
            ]
            q_offset = past_key_values.get_seq_length(self.layer_idx)

        use_attn_mask = (
            attention_mask is not None and not attention_mask.all() and (n > 1)
        )
        if use_attn_mask:
            start = q_offset
            attention_mask_ = attention_mask[:, start:].unsqueeze(-1).unsqueeze(-1)
            k = k.masked_fill(attention_mask_ == 0, 0)
            log_f = log_f.masked_fill(attention_mask_.squeeze(-1) == 0, 0)

        if self.causal:
            if (self.training or use_cache) and (v.shape[1] > 1):
                output, recurrent_state_kk, recurrent_state_kv = chunk_mesa_net(
                    q=q,
                    k=k,
                    v=v,
                    g=log_f,
                    beta=beta,
                    lamb=lamb,
                    output_final_state=True,
                    max_CG_iteration=self.max_cg_step_training,
                )
            else:
                (
                    output,
                    recurrent_state_kk,
                    recurrent_state_kv,
                ) = mesa_net_decoding_one_step(
                    q=q.squeeze(1),
                    k=k.squeeze(1),
                    v=v.squeeze(1),
                    g=log_f.squeeze(1),
                    beta=beta.squeeze(1),
                    lamb=lamb,
                    prev_h_kk=recurrent_state_kk,
                    prev_h_kv=recurrent_state_kv,
                    max_CG_iteration=self.max_cg_step_decoding,
                )
                output = output.unsqueeze(1).to(q)
        else:
            assert False

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=[recurrent_state_kk, recurrent_state_kv],
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
