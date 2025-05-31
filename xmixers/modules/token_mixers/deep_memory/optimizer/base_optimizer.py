from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from .utils import get_pooling_fn


class FastWeightOptimizer(ABC):
    """
    Abstract base class for custom optimizers.

    This class provides common functionality for optimizers including
    parameter management, gradient zeroing, and pooling operations.
    """

    @abstractmethod
    def __init__(
        self, fast_weight: Dict[str, torch.nn.Parameter], pooling_type: str = "mean"
    ):
        """
        Initialize the optimizer.

        Args:
            fast_weight: Dictionary mapping parameter names to parameters
            pooling_type: Type of pooling to use for tensor operations
        """
        self.fast_weight = fast_weight
        self.pooling_fn = get_pooling_fn(pooling_type)
        self.state: Dict[str, Dict[str, Any]] = {}

    def setup_state(self) -> None:
        """Initialize state dictionaries for all parameters."""
        self.state = {}
        for name in self.params_dict.keys():
            self.state[name] = {}

    def zero_grad(self) -> None:
        """Zero out gradients for all parameters."""
        for param in self.params_dict.values():
            if param.grad is not None:
                param.grad.zero_()

    def pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling operation to tensor if it has multiple elements in dimension 1.

        Args:
            x: Input tensor

        Returns:
            Pooled tensor or original tensor if pooling not applicable
        """
        if isinstance(x, torch.Tensor) and x.shape[1] > 1:
            return self.pooling_fn(x)
        else:
            return x

    @abstractmethod
    def step(
        self,
        lr_dict: Optional[Dict[str, float]] = None,
        wd_dict: Optional[Dict[str, float]] = None,
        momentum_dict: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Perform a single optimization step.

        Args:
            lr_dict: Dictionary of learning rates per parameter
            wd_dict: Dictionary of weight decay values per parameter
            momentum_dict: Dictionary of momentum values per parameter
        """
