from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_ttt import TTTConfig
from .modeling_ttt import TTTForCausalLM, TTTLayer, TTTModel

AutoConfig.register(TTTConfig.model_type, TTTConfig)
AutoModel.register(TTTConfig, TTTModel)
AutoModelForCausalLM.register(TTTConfig, TTTForCausalLM)

__all__ = [
    "TTTConfig",
    "TTTModel",
    "TTTForCausalLM",
]
