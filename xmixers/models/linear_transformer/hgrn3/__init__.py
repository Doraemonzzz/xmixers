from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_linear_transformer import LinearTransformerConfig
from .modeling_linear_transformer import (
    LinearTransformerForCausalLM,
    LinearTransformerLayer,
    LinearTransformerModel,
)

AutoConfig.register(LinearTransformerConfig.model_type, LinearTransformerConfig)
AutoModel.register(LinearTransformerConfig, LinearTransformerModel)
AutoModelForCausalLM.register(LinearTransformerConfig, LinearTransformerForCausalLM)

__all__ = [
    "LinearTransformerConfig",
    "LinearTransformerModel",
    "LinearTransformerForCausalLM",
]
