from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_tnl import TnlConfig
from .modeling_tnl import TnlForCausalLM, TnlLayer, TnlModel

AutoConfig.register(TnlConfig.model_type, TnlConfig)
AutoModel.register(TnlConfig, TnlModel)
AutoModelForCausalLM.register(TnlConfig, TnlForCausalLM)

__all__ = [
    "TnlConfig",
    "TnlModel",
    "TnlForCausalLM",
]
