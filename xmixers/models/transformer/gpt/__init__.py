from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_gpt import GPTConfig
from .modeling_gpt import GPTForCausalLM, GPTLayer, GPTModel

AutoConfig.register(GPTConfig.model_type, GPTConfig)
AutoModel.register(GPTConfig, GPTModel)
AutoModelForCausalLM.register(GPTConfig, GPTForCausalLM)

__all__ = ["GPTConfig", "GPTModel", "GPTForCausalLM"]
