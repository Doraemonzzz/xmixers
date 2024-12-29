from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_naive_hybrid import NaiveHybridConfig
from .modeling_naive_hybrid import (
    NaiveHybridForCausalLM,
    NaiveHybridLayer,
    NaiveHybridModel,
)

AutoConfig.register(NaiveHybridConfig.model_type, NaiveHybridConfig)
AutoModel.register(NaiveHybridConfig, NaiveHybridModel)
AutoModelForCausalLM.register(NaiveHybridConfig, NaiveHybridForCausalLM)

__all__ = ["NaiveHybridConfig", "NaiveHybridModel", "NaiveHybridForCausalLM"]
