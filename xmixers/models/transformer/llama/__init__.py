from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_llama import LlamaConfig
from .modeling_llama import LlamaForCausalLM, LlamaModel

AutoConfig.register(LlamaConfig.model_type, LlamaConfig)
AutoModel.register(LlamaConfig, LlamaModel)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM)

__all__ = ["LlamaConfig", "LlamaModel", "LlamaForCausalLM"]
