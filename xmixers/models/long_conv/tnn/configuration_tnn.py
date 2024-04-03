# coding=utf-8
""" Tnn configuration"""

from typing import List

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class TnnConfig(PretrainedConfig):
    model_type = "tnn"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        vocab_size: int = 50272,
        use_cache: bool = True,
        init_std: float = 0.02,
        ##### model config
        # gtu config
        embed_dim: int = 768,
        expand_ratio: int = 1,
        bias: bool = False,
        gtu_activation: str = "silu",
        causal: bool = True,
        norm_type: str = "layernorm",
        use_decay: bool = True,
        rpe_in_dim: int = 1,
        rpe_feature_dim: int = 32,  # for rpe in tno
        rpe_layers: int = 3,
        dims: List[int] = [-2],
        # glu config
        mid_dim: int = 1024,
        glu_activation: str = "silu",
        # others
        num_layers: int = 24,
        add_bos_token: bool = False,
        max_position_embeddings: int = 2048,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # hf origin
        self.vocab_size = vocab_size
        self.use_cache = use_cache
        self.init_std = init_std
        # add
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.bias = bias

        self.gtu_activation = gtu_activation
        self.causal = causal
        self.norm_type = norm_type
        self.use_decay = use_decay
        self.rpe_in_dim = rpe_in_dim
        self.rpe_feature_dim = rpe_feature_dim
        self.rpe_layers = rpe_layers
        self.dims = dims
        # glu config
        self.mid_dim = mid_dim
        self.glu_activation = glu_activation
        # others
        self.num_layers = num_layers
        self.add_bos_token = add_bos_token
        self.max_position_embeddings = max_position_embeddings
