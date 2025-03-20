from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_gsa import GsaConfig
from .modeling_gsa import GsaForCausalLM, GsaLayer, GsaModel

AutoConfig.register(GsaConfig.model_type, GsaConfig)
AutoModel.register(GsaConfig, GsaModel)
AutoModelForCausalLM.register(GsaConfig, GsaForCausalLM)

__all__ = [
    "GsaConfig",
    "GsaModel",
    "GsaForCausalLM",
]
