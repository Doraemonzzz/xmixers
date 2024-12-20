# coding=utf-8
""" Llama mpa configuration"""

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LlamaMpaConfig(LlamaConfig):
    model_type = "llama_mpa"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
