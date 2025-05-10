from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mamba2 import Mamba2XmixersConfig
from .modeling_mamba2 import Mamba2ForCausalLM, Mamba2Layer, Mamba2Model

AutoConfig.register(Mamba2XmixersConfig.model_type, Mamba2XmixersConfig)
AutoModel.register(Mamba2XmixersConfig, Mamba2Model)
AutoModelForCausalLM.register(Mamba2XmixersConfig, Mamba2ForCausalLM)

__all__ = [
    "Mamba2XmixersConfig",
    "Mamba2Model",
    "Mamba2ForCausalLM",
]
