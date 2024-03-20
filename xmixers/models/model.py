import torch.nn as nn


class SequenceModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        n_layer = kwargs.get("n_layer", default=1)

    def get_block_config(self, **kwargs):
        pass

    def build_block(self, **kwargs):
        pass
