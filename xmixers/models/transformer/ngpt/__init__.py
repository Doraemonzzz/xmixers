from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_ngpt import nGPTConfig
from .modeling_ngpt import nGPTForCausalLM, nGPTLayer, nGPTModel

AutoConfig.register(nGPTConfig.model_type, nGPTConfig)
AutoModel.register(nGPTConfig, nGPTModel)
AutoModelForCausalLM.register(nGPTConfig, nGPTForCausalLM)

__all__ = ["nGPTConfig", "nGPTModel", "nGPTForCausalLM"]
