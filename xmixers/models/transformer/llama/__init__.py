from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_llama import LLaMAConfig
from .modeling_llama import LLaMAForCausalLM, LLaMALayer, LLaMAModel

AutoConfig.register(LLaMAConfig.model_type, LLaMAConfig)
AutoModel.register(LLaMAConfig, LLaMAModel)
AutoModelForCausalLM.register(LLaMAConfig, LLaMAForCausalLM)

__all__ = ["LLaMAConfig", "LLaMAModel", "LLaMAForCausalLM"]
