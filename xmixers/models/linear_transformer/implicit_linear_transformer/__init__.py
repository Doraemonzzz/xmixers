from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_implicit_linear_transformer import ImplicitLinearTransformerConfig
from .modeling_implicit_linear_transformer import (
    ImplicitLinearTransformerForCausalLM,
    ImplicitLinearTransformerLayer,
    ImplicitLinearTransformerModel,
)

AutoConfig.register(
    ImplicitLinearTransformerConfig.model_type, ImplicitLinearTransformerConfig
)
AutoModel.register(ImplicitLinearTransformerConfig, ImplicitLinearTransformerModel)
AutoModelForCausalLM.register(
    ImplicitLinearTransformerConfig, ImplicitLinearTransformerForCausalLM
)

__all__ = [
    "ImplicitLinearTransformerConfig",
    "ImplicitLinearTransformerModel",
    "ImplicitLinearTransformerForCausalLM",
]
