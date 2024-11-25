from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_hgrn3 import Hgrn3Config
from .modeling_hgrn3 import Hgrn3ForCausalLM, Hgrn3Layer, Hgrn3Model

AutoConfig.register(Hgrn3Config.model_type, Hgrn3Config)
AutoModel.register(Hgrn3Config, Hgrn3Model)
AutoModelForCausalLM.register(Hgrn3Config, Hgrn3ForCausalLM)

__all__ = [
    "Hgrn3Config",
    "Hgrn3Model",
    "Hgrn3ForCausalLM",
]
