import math

import torch
import torch.nn as nn

from .constants import EMBED_DIM_BASE


# for token mixer
def _initialize_weights(self, module):
    if getattr(module, "_is_hf_initialized", False):
        return

    if self.token_mixer_init_type == 0:
        return
    elif self.token_mixer_init_type in [1, 2, 3, 4]:
        if self.token_mixer_init_type == 1:  # fla init
            gain = 2**-2.5
        elif self.token_mixer_init_type == 2:  # fairseq init
            gain = 2**-0.5
        elif self.token_mixer_init_type == 3:  # minicpm init
            gain = self.init_std / ((self.embed_dim / EMBED_DIM_BASE) ** 0.5)
        elif self.token_mixer_init_type == 4:  # for test
            gain = self.gain

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if hasattr(module, "q"):
            nn.init.ones_(module.q)

        if hasattr(module, "k"):
            nn.init.ones_(module.k)

        if hasattr(module, "log_decay"):
            nn.init.zeros_(module.log_decay)

        if hasattr(module, "k_head"):
            nn.init.xavier_uniform_(module.k_head, gain=gain)

        if hasattr(module, "v_head"):
            nn.init.xavier_uniform_(module.v_head, gain=gain)

        if hasattr(module, "state") and module.state is not None:
            nn.init.zeros_(module.state)

        if hasattr(module, "scale") and module.scale is not None:
            nn.init.ones_(module.scale)

        if hasattr(module, "initial_state") and module.initial_state is not None:
            nn.init.zeros_(module.initial_state)

        if (
            hasattr(module, "initial_state_bias")
            and module.initial_state_bias is not None
        ):
            nn.init.zeros_(module.initial_state_bias)

        # ttt
        if hasattr(module, "ln_weight") and module.ln_weight is not None:
            nn.init.ones_(module.ln_weight)

        if hasattr(module, "ln_bias") and module.ln_bias is not None:
            nn.init.zeros_(module.ln_bias)

    if self.rescale_type == 1:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference: https://github.com/karpathy/nanoGPT/blob/master/model.py#L144 https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/models/gla/modeling_gla.py#L152
        for name, p in module.named_parameters():
            if name in ["o_proj.weight", "out_proj.weight"]:
                num_residuals_per_layer = 2
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.num_layers)
    elif self.rescale_type == 2:
        for name, p in module.named_parameters():
            if name in ["o_proj.weight", "out_proj.weight"]:
                with torch.no_grad():
                    p *= 0

    module._is_hf_initialized = True


# for PreTrainedModel
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
                # torchtitan has bug here
                try:
                    nn.init.zeros_(module.weight[module.padding_idx])
                except:
                    pass
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()
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
                # torchtitan has bug here
                try:
                    nn.init.zeros_(module.weight[module.padding_idx])
                except:
                    pass
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()
    elif (
        self.config.init_type == 2
    ):  # credit to https://arxiv.org/pdf/1910.05895 and https://arxiv.org/pdf/2405.04434
        std = min((2 / 5 / self.config.embed_dim) ** 0.5, 0.006)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                # torchtitan has bug here
                try:
                    nn.init.zeros_(module.weight[module.padding_idx])
                except:
                    pass
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

    # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
    #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
    #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
    #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
    #
    # Reference: https://github.com/karpathy/nanoGPT/blob/master/model.py#L144 https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/models/gla/modeling_gla.py#L152
    for name, p in module.named_parameters():
        if name in ["out_proj.weight", "w3.weight"]:
            num_residuals_per_layer = 2
            # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
            # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
            # We need to reinit p since this code could be called multiple times
            # Having just p *= scale would repeatedly scale it down
            with torch.no_grad():
                p /= math.sqrt(num_residuals_per_layer * self.config.num_layers)

            if self.config.rescale_type == 2:
                with torch.no_grad():
                    p *= 0

    module._is_hf_initialized = True


def _post_init_weights(
    self,
):
    # reset the _is_hf_initialized to False
    self.apply(lambda m: setattr(m, "_is_hf_initialized", False))
    # Create an unbound function that takes module as argument
    init_fn = lambda module: _init_weights(self, module)
    self.model.embed_tokens.apply(init_fn)
    self.model.final_norm._init_weights()
    if hasattr(self.model, "tpe") and self.model.tpe is not None:
        self.model.tpe._init_weights()
        self.model.tpe.norm._init_weights()
    for layer in self.model.layers:
        # for token mixer, use custom init first
        layer.token_mixer._init_weights()
        if hasattr(layer.token_mixer, "lrpe"):
            layer.token_mixer.lrpe._init_weights()
        # if not using custom init, use default init
        layer.token_mixer.apply(init_fn)
        layer.channel_mixer.apply(init_fn)
        layer.token_norm._init_weights()
        layer.channel_norm._init_weights()

    if self.config.tie_word_embeddings:
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())
    else:
        self.lm_head.apply(init_fn)
