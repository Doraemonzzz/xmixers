from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import FusedLinearCrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    apply_rotary_pos_emb,
)

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None


class LlamaMpaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.kv_proj = nn.Linear(
            config.hidden_size, 2 * self.head_dim, bias=config.attention_bias
        )
        self.kv_head_proj = nn.Linear(
            config.hidden_size,
            2 * config.num_attention_heads,
            bias=config.attention_bias,
        )
        self.act = F.sigmoid
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # x: b n d
        # linear map
        q = self.q_proj(hidden_states)
        k, v = self.kv_proj(hidden_states).chunk(2, dim=-1)
        k_head, v_head = self.kv_head_proj(hidden_states).chunk(2, dim=-1)

        k_head = self.act(k_head)
        v_head = self.act(v_head)

        k, v = map(
            lambda arr: torch.einsum("... d, ... h -> ... h d", arr[0], arr[1]),
            [(k, k_head), (v, v_head)],
        )
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)

        query_states, key_states, value_states = map(
            lambda x: rearrange(x, "... n h d -> ... h n d"),
            [q, k, v],
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = None
        query_states, key_states, value_states = map(
            lambda x: rearrange(x.to(value_states.dtype), "b h n d -> b n h d"),
            [query_states, key_states, value_states],
        )
        attn_output = flash_attn_func(
            query_states, key_states, value_states, softmax_scale=self.scaling
        )
        attn_output = rearrange(attn_output, "b n h d -> b n (h d)")

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class LlamaMpaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = LlamaMpaAttention(config, layer_idx)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.training
        logits = (
            None
            if fuse_linear_and_cross_entropy
            else self.lm_head(hidden_states[:, -num_logits_to_keep:])
        )

        loss = None
        if labels is not None:
            if fuse_linear_and_cross_entropy:
                loss_fct = FusedLinearCrossEntropyLoss()
            else:
                loss_fct = nn.CrossEntropyLoss()
            # Enable model parallelism
            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (
                    labels[..., 1:],
                    torch.full_like(labels[:, :1], loss_fct.ignore_index),
                ),
                1,
            )
            if fuse_linear_and_cross_entropy:
                loss = loss_fct(
                    hidden_states.view(-1, self.config.hidden_size),
                    labels.view(-1),
                    self.lm_head.weight,
                    self.lm_head.bias,
                )
            else:
                loss = loss_fct(
                    logits.view(-1, self.config.vocab_size), labels.view(-1)
                )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
