from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_metala import MetaLaConfig
from .modeling_metala import MetaLaForCausalLM, MetaLaLayer, MetaLaModel

AutoConfig.register(MetaLaConfig.model_type, MetaLaConfig)
AutoModel.register(MetaLaConfig, MetaLaModel)
AutoModelForCausalLM.register(MetaLaConfig, MetaLaForCausalLM)

__all__ = [
    "MetaLaConfig",
    "MetaLaModel",
    "MetaLaForCausalLM",
]
