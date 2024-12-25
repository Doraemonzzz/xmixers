from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_hgrn2 import Hgrn2Config
from .modeling_hgrn2 import Hgrn2ForCausalLM, Hgrn2Layer, Hgrn2Model

AutoConfig.register(Hgrn2Config.model_type, Hgrn2Config)
AutoModel.register(Hgrn2Config, Hgrn2Model)
AutoModelForCausalLM.register(Hgrn2Config, Hgrn2ForCausalLM)

__all__ = [
    "Hgrn2Config",
    "Hgrn2Model",
    "Hgrn2ForCausalLM",
]
