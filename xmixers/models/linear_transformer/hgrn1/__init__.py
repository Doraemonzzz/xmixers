from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_hgrn1 import Hgrn1Config
from .modeling_hgrn1 import Hgrn1ForCausalLM, Hgrn1Layer, Hgrn1Model

AutoConfig.register(Hgrn1Config.model_type, Hgrn1Config)
AutoModel.register(Hgrn1Config, Hgrn1Model)
AutoModelForCausalLM.register(Hgrn1Config, Hgrn1ForCausalLM)

__all__ = [
    "Hgrn1Config",
    "Hgrn1Model",
    "Hgrn1ForCausalLM",
]
