from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_lightnet import LightNetConfig
from .modeling_lightnet import LightNetForCausalLM, LightNetLayer, LightNetModel

AutoConfig.register(LightNetConfig.model_type, LightNetConfig)
AutoModel.register(LightNetConfig, LightNetModel)
AutoModelForCausalLM.register(LightNetConfig, LightNetForCausalLM)

__all__ = [
    "LightNetConfig",
    "LightNetModel",
    "LightNetForCausalLM",
]
