from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_lm_head_hybrid import LmHeadHybridConfig
from .modeling_lm_head_hybrid import (
    LmHeadHybridForCausalLM,
    LmHeadHybridLayer,
    LmHeadHybridModel,
)

AutoConfig.register(LmHeadHybridConfig.model_type, LmHeadHybridConfig)
AutoModel.register(LmHeadHybridConfig, LmHeadHybridModel)
AutoModelForCausalLM.register(LmHeadHybridConfig, LmHeadHybridForCausalLM)

__all__ = ["LmHeadHybridConfig", "LmHeadHybridModel", "LmHeadHybridForCausalLM"]
