import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5, **kwargs):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        # TODO: add a fusion here
        return torch.tanh(self.alpha * x) * self.weight

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}"


class DynamicTanhFusedGate(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5, **kwargs):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x, gate):
        # TODO: add a fusion here
        return torch.tanh(self.alpha * x) * self.weight * gate

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}"
