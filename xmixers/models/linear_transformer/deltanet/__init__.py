from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_deltanet import DeltaNetConfig
from .modeling_deltanet import DeltaNetForCausalLM, DeltaNetLayer, DeltaNetModel

AutoConfig.register(DeltaNetConfig.model_type, DeltaNetConfig)
AutoModel.register(DeltaNetConfig, DeltaNetModel)
AutoModelForCausalLM.register(DeltaNetConfig, DeltaNetForCausalLM)

__all__ = [
    "DeltaNetConfig",
    "DeltaNetModel",
    "DeltaNetForCausalLM",
]
