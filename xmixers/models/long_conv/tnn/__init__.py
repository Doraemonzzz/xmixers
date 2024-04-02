from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_tnn import TnnConfig
from .modeling_tnn import TnnForCausalLM, TnnModel

AutoConfig.register(TnnConfig.model_type, TnnConfig)
AutoModel.register(TnnConfig, TnnModel)
AutoModelForCausalLM.register(TnnModel, TnnForCausalLM)

__all__ = ["TnnConfig", "TnnModel", "TnnForCausalLM"]
