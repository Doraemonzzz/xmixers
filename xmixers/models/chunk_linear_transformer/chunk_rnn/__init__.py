from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_chunk_rnn import ChunkRnnConfig
from .modeling_chunk_rnn import ChunkRnnForCausalLM, ChunkRnnLayer, ChunkRnnModel

AutoConfig.register(ChunkRnnConfig.model_type, ChunkRnnConfig)
AutoModel.register(ChunkRnnConfig, ChunkRnnModel)
AutoModelForCausalLM.register(ChunkRnnConfig, ChunkRnnForCausalLM)

__all__ = [
    "ChunkRnnConfig",
    "ChunkRnnModel",
    "ChunkRnnForCausalLM",
]
