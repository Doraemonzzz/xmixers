from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_llama_mpa import LlamaMpaConfig
from .modeling_llama_mpa import LlamaMpaAttention, LlamaMpaForCausalLM

AutoConfig.register(LlamaMpaConfig.model_type, LlamaMpaConfig)
AutoModelForCausalLM.register(LlamaMpaConfig, LlamaMpaForCausalLM)

__all__ = ["LlamaMpaConfig", "LlamaMpaForCausalLM"]
