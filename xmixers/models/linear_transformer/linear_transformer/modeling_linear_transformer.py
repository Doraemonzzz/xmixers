# coding=utf-8
""" PyTorch LinearTransformer model."""
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


from xmixers.modules import get_channel_mixer, get_norm_fn, get_token_mixer
from xmixers.modules.pes import LearnablePe, SinCosPe, Tpe
from xmixers.utils import XmixersCache, _init_weights, _post_init_weights, pad_embed_dim
from xmixers.utils.loss_utils import Loss

from .configuration_linear_transformer import LinearTransformerConfig


class LinearTransformerLayer(nn.Module):
    def __init__(self, config: LinearTransformerConfig, layer_idx=0):
        super().__init__()

        self.token_mixer = get_token_mixer(config, layer_idx)
        self.token_norm = get_norm_fn(config.norm_type)(
            config.embed_dim, bias=config.bias
        )
        self.channel_mixer = get_channel_mixer(config)
        self.channel_norm = get_norm_fn(config.norm_type)(
            config.embed_dim, bias=config.bias
        )

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        # token mixer
        residual = x
        x, past_key_values = self.token_mixer(
            x=self.token_norm(x),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        x = x + residual

        # channel mixer
        x = self.channel_mixer(self.channel_norm(x)) + x

        outputs = (x, past_key_values)

        return outputs


class LinearTransformerPreTrainedModel(PreTrainedModel):
    config_class = LinearTransformerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["LinearTransformerLayer"]

    def _init_weights(self, module):
        return _init_weights(self, module)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LinearTransformerModel):
            module.gradient_checkpointing = value


class LinearTransformerModel(LinearTransformerPreTrainedModel):
    def __init__(self, config: LinearTransformerConfig):
        super().__init__(config)
        # hf origin
        self.padding_idx = config.pad_token_id
        if hasattr(config, "pad_embed_dim") and config.pad_embed_dim:
            config.vocab_size = pad_embed_dim(config.vocab_size)
        self.config = config
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        # params
        self.use_tpe = config.use_tpe
        if self.use_tpe:
            self.tpe = Tpe(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                bias=False,
                layer_idx=0,
                token_mixer_norm_type=config.token_mixer_norm_type,
            )
            offset = 1
        else:
            offset = 0
        self.embed_scale = config.embed_dim**0.5 if config.use_embed_scale else 1
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.embed_dim, self.padding_idx
        )
        self.use_ape = config.use_ape
        if self.use_ape:
            if config.ape_type == "sincos":
                self.ape = SinCosPe(config.embed_dim, config.base)
            else:
                self.ape = LearnablePe(config.embed_dim, config.max_position_embeddings)

        self.layers = nn.ModuleList(
            [
                LinearTransformerLayer(config, layer_idx + offset)
                for layer_idx in range(config.num_layers)
            ]
        )
        self.final_norm = get_norm_fn(config.norm_type)(
            config.embed_dim, bias=config.bias
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.use_cache if not self.training else False)
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if use_cache and not isinstance(past_key_values, XmixersCache):
            past_key_values = XmixersCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.embed_scale * inputs_embeds

        if self.use_ape:
            # get offset
            offset = 0
            if past_key_values is not None:
                offset = past_key_values.get_seq_length(0)
            hidden_states = self.ape(hidden_states, offset=offset)

        if self.use_tpe:
            hidden_states, past_key_values = self.tpe(
                x=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    **kwargs,
                )
            else:
                hidden_states, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )

        hidden_states = self.final_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LinearTransformerForCausalLM(LinearTransformerPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LinearTransformerModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)
        self.loss = Loss()
        # Initialize weights and apply final processing
        self.post_init()

    def post_init_weights(
        self,
    ):
        _post_init_weights(self)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LinearTransformerForCausalLM

        >>> model = LinearTransformerForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
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
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        fuse_linear_and_cross_entropy = (
            self.config.ce_type not in ["fla_fce", "naive", "xopes_ce"]
            and self.training
        )

        # since we may use forward_pre_hook, we can't use key word arguments
        logits, loss = self.loss(
            self.lm_head.weight,
            hidden_states,
            labels,
            self.lm_head.bias,
            num_logits_to_keep,
            fuse_linear_and_cross_entropy,
            ce_type=self.config.ce_type,
        )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # only last token for `inputs_ids` if the `past_key_values` is passed along.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past
