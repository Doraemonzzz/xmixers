# coding=utf-8
# nGLU: https://arxiv.org/pdf/2410.01131
""" PyTorch nGPT model."""
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


from xmixers.modules import LearnablePe, nAttention, nGLU
from xmixers.utils import endswith, print_module

from .configuration_ngpt import nGPTConfig


class nGPTLayer(nn.Module):
    def __init__(self, config: nGPTConfig, layer_idx=0):
        super().__init__()

        self.token_mixer = nAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            kv_heads=config.kv_heads,
            bias=False,
            use_lrpe=config.use_lrpe,
            layer_idx=layer_idx,
            base=config.base,
            token_mixer_init_type=config.token_mixer_init_type,
            rescale_type=config.rescale_type,
            num_layers=config.num_layers,
            init_std=config.init_std,
        )
        base_scale = 1.0 / (config.embed_dim**0.5)
        self.token_alpha_init_value = 0.05
        self.token_alpha_init_scaling = base_scale
        self.token_alpha = torch.nn.Parameter(
            self.token_alpha_init_scaling
            * torch.ones(config.embed_dim, dtype=torch.float32)
        )

        self.channel_mixer = nGLU(
            embed_dim=config.embed_dim,
            mid_dim=config.mid_dim,
            activation=config.glu_activation,
            bias=False,
        )
        self.channel_alpha_init_value = 0.05
        self.channel_alpha_init_scaling = base_scale
        self.channel_alpha = torch.nn.Parameter(
            self.channel_alpha_init_scaling
            * torch.ones(config.embed_dim, dtype=torch.float32)
        )

    def extra_repr(self):
        return print_module(self)

    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(
        self,
        x,
        attention_mask: Optional[torch.Tensor] = None,  # (b, m)
        past_key_values: Optional[Cache] = None,
    ):
        # token mixer
        residual = x
        x, past_key_values = self.token_mixer(
            x=x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        # residual
        lr = self.token_alpha * (
            self.token_alpha_init_value / self.token_alpha_init_scaling
        )
        lr = torch.abs(lr)

        A_norm = self.justnorm(residual)  # normally, normalization is not needed
        B_norm = self.justnorm(x)

        x = A_norm + lr * (B_norm - A_norm)
        x = self.justnorm(x)

        # channel mixer
        residual = x
        x = self.channel_mixer(x)
        # residual
        lr = self.channel_alpha * (
            self.channel_alpha_init_value / self.channel_alpha_init_scaling
        )
        lr = torch.abs(lr)

        A_norm = self.justnorm(residual)  # normally, normalization is not needed
        B_norm = self.justnorm(x)

        x = A_norm + lr * (B_norm - A_norm)
        x = self.justnorm(x)

        outputs = (x, past_key_values)

        return outputs


class nGPTPreTrainedModel(PreTrainedModel):
    config_class = nGPTConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["nGPTLayer"]

    def extra_repr(self):
        return print_module(self)

    def justnorm(self, x, idim=-1):
        dtype = x.dtype
        x = x.float()
        res = (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype)
        return res

    def _init_weights(self, module):
        if self.config.init_type == 0:
            std = self.config.init_std
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, LearnablePe):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        elif (
            self.config.init_type == 1
        ):  # credit to https://arxiv.org/pdf/2409.02060#page=14.84
            std = self.config.init_std
            trunc_std = 3 * std
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight, mean=0.0, std=std, a=-trunc_std, b=trunc_std
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(
                    module.weight, mean=0.0, std=std, a=-trunc_std, b=trunc_std
                )
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, LearnablePe):
                nn.init.trunc_normal_(
                    module.weight, mean=0.0, std=std, a=-trunc_std, b=trunc_std
                )
        elif self.config.init_type == 2:  # credit to https://arxiv.org/pdf/1910.05895
            std = (2 / 5 / self.config.embed_dim) ** 0.5
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, LearnablePe):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Reinitialize selected weights subject to the OpenAI nGPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- nGPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference: https://github.com/karpathy/nanonGPT/blob/master/model.py#L144 https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/models/gla/modeling_gla.py#L152
        for name, p in module.named_parameters():
            if name in ["w3.weight"]:
                num_residuals_per_layer = 2
                # module.weight.data.normal_(mean=0.0, std=std/math.sqrt(2 * self.config.num_layers))
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_layers)

        # nGPT post init
        for name, p in module.named_parameters():
            if not name.endswith("weight"):
                continue
            if endswith(name, ["out_proj.weight", "w3.weight"]):
                p = self.justnorm(p, idim=0)
            else:
                p = self.justnorm(p, idim=1)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, nGPTModel):
            module.gradient_checkpointing = value


class nGPTModel(nGPTPreTrainedModel):
    def __init__(self, config: nGPTConfig):
        super().__init__(config)
        # hf origin
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        # params
        self.embed_scale = config.embed_dim**0.5 if config.use_embed_scale else 1
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.embed_dim, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [nGPTLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
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

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.embed_scale * inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                layer_outputs[-1]

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class nGPTForCausalLM(nGPTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = nGPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)
        # final norm
        base_scale = 1.0 / (config.embed_dim**0.5)
        self.sz_init_value = 1.00
        self.sz_init_scaling = base_scale
        self.sz = torch.nn.Parameter(
            self.sz_init_scaling * torch.ones(config.vocab_size, dtype=torch.float32)
        )

        # Initialize weights and apply final processing
        self.post_init()

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
        >>> from transformers import AutoTokenizer, nGPTForCausalLM

        >>> model = nGPTForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
        logits = sz * logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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
