from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.simple_gla import chunk_simple_gla, fused_recurrent_simple_gla
from transformers.cache_utils import Cache

from xmixers.modules.activations import get_activation_fn
from xmixers.modules.normalizations import get_norm_fn
from xmixers.utils import XMIXERS_DEBUG, print_params


def l2_norm(x, eps):
    return F.normalize(x, p=2.0, dim=-1, eps=eps)


class Hgru3(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expand_ratio: int,
        bias: bool = False,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        norm_type: str = "layernorm",
        q_activation: str = "silu",
        k_activation: str = "silu",
        beta_activation: str = "silu",
        causal: bool = True,
        use_dense_memory: bool = True,
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
        self.expand_ratio = expand_ratio
        self.causal = causal
        self.use_dense_memory = use_dense_memory
        self.use_output_gate = use_output_gate

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.f_proj = nn.Linear(embed_dim, embed_dim // expand_ratio, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_act = get_activation_fn(q_activation)
        self.k_act = get_activation_fn(k_activation)
        self.beta_act = get_activation_fn(beta_activation)
        self.norm = get_norm_fn(norm_type)(embed_dim)

        if self.use_dense_memory:
            # !!! dont use beta as name in hf: https://github.com/huggingface/transformers/issues/29554
            # I - 2 * beta beta ^ T
            self.bet_proj = nn.Sequential(
                nn.Linear(embed_dim, expand_ratio, bias=bias),
                nn.Linear(expand_ratio, embed_dim, bias=bias),
            )

        if self.use_output_gate:
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, expand_ratio, bias=bias),
                nn.Linear(expand_ratio, embed_dim, bias=bias),
            )

    def merge_log_decay(self, log_decay, need_repeat=True):
        dtype = log_decay.dtype
        d = log_decay.shape[-1]
        # b h n d -> b h n -> b h n d
        if self.mean_type == 1:  # 算数平均
            decay = torch.exp(log_decay.float()).mean(dim=-1)
            log_decay_new = torch.log(decay)
        else:  # 几何平均
            log_decay_new = log_decay.float().mean(dim=-1)

        if need_repeat:
            log_decay_new = repeat(log_decay_new, "... -> ... d", d=d)

        return log_decay_new.to(dtype)

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        log_lower_bound: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # x: b n d
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        f = self.f_proj(x)

        # act
        q = self.q_act(q)
        k = self.k_act(k)

        # dense memory
        # todo: make a fusion here
        if self.use_dense_memory:
            # I - 2 beta beta ^ T
            beta = self.beta_act(self.bet_proj(x))
            beta = l2_norm(beta, 1e-6).contiguous()
            q_beta = (q * beta).sum(dim=-1, keepdim=True)
            q = q - 2 * q_beta * beta

        # h is num_head, d is head dimension
        q, k, v = map(
            lambda x: rearrange(x, "... (h d) -> ... h d", d=self.expand_ratio),
            [q, k, v],
        )

        # todo: make a fusion here
        # l + (1 - l) * sigmoid(x)
        lower_bound = torch.exp(log_lower_bound.float())
        f = lower_bound + (1 - lower_bound) * F.sigmoid(f.float())
        log_f = torch.log(f)

        q, k, v, log_f = map(
            lambda x: rearrange(x, "b n h ... -> b h n ...").contiguous(),
            [q, k, v, log_f],
        )

        recurrent_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]["recurrent_state"]

        if self.causal:
            if self.training:
                fn = chunk_simple_gla
            else:
                fn = fused_recurrent_simple_gla

            output, recurrent_state = fn(
                q=q,
                k=k,
                v=v,
                g=log_f,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        else:
            assert False

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx,
                offset=x.shape[-2],
            )

        output = rearrange(output, "b h n d -> b n (h d)")

        if self.use_output_gate:
            output_gate = F.sigmoid(self.output_gate(x))
            output = output * output_gate

        # out proj
        output = self.out_proj(output)

        # use post norm here for better parallel when using tp
        output = self.norm(output)

        return output, past_key_values
