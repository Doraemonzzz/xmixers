from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_flex_gpt import FlexGPTConfig
from .modeling_flex_gpt import FlexGPTForCausalLM, FlexGPTLayer, FlexGPTModel

AutoConfig.register(FlexGPTConfig.model_type, FlexGPTConfig)
AutoModel.register(FlexGPTConfig, FlexGPTModel)
AutoModelForCausalLM.register(FlexGPTConfig, FlexGPTForCausalLM)

__all__ = ["FlexGPTConfig", "FlexGPTModel", "FlexGPTForCausalLM"]
