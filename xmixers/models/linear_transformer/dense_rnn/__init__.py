from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_dense_rnn import DenseRnnConfig
from .modeling_dense_rnn import DenseRnnForCausalLM, DenseRnnLayer, DenseRnnModel

AutoConfig.register(DenseRnnConfig.model_type, DenseRnnConfig)
AutoModel.register(DenseRnnConfig, DenseRnnModel)
AutoModelForCausalLM.register(DenseRnnConfig, DenseRnnForCausalLM)

__all__ = [
    "DenseRnnConfig",
    "DenseRnnModel",
    "DenseRnnForCausalLM",
]
