from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_decay_linear_transformer import DecayLinearTransformerConfig
from .modeling_decay_linear_transformer import (
    DecayLinearTransformerForCausalLM,
    DecayLinearTransformerLayer,
    DecayLinearTransformerModel,
)

AutoConfig.register(
    DecayLinearTransformerConfig.model_type, DecayLinearTransformerConfig
)
AutoModel.register(DecayLinearTransformerConfig, DecayLinearTransformerModel)
AutoModelForCausalLM.register(
    DecayLinearTransformerConfig, DecayLinearTransformerForCausalLM
)

__all__ = [
    "DecayLinearTransformerConfig",
    "DecayLinearTransformerModel",
    "DecayLinearTransformerForCausalLM",
]
