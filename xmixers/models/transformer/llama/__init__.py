from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_llama import LLaMAConfig
from .modeling_llama import LlamaForCausalLM, LlamaModel

AutoConfig.register(LLaMAConfig.model_type, LLaMAConfig)
AutoModel.register(LLaMAConfig, LlamaModel)
AutoModelForCausalLM.register(LLaMAConfig, LlamaForCausalLM)

__all__ = ["LLaMAConfig", "LlamaModel", "LlamaForCausalLM"]
