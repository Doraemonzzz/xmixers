from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_polar_rnn import PolarRnnConfig
from .modeling_polar_rnn import PolarRnnForCausalLM, PolarRnnLayer, PolarRnnModel

AutoConfig.register(PolarRnnConfig.model_type, PolarRnnConfig)
AutoModel.register(PolarRnnConfig, PolarRnnModel)
AutoModelForCausalLM.register(PolarRnnConfig, PolarRnnForCausalLM)

__all__ = [
    "PolarRnnConfig",
    "PolarRnnModel",
    "PolarRnnForCausalLM",
]
