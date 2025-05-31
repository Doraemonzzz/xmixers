from typing import Dict

import torch

from .base_optimizer import FastWeightOptimizer
from .utils import get_pooling_fn


class SGD(FastWeightOptimizer):
    def __init__(
        self,
        fast_weight: Dict[str, torch.Tensor],
        lr: float = 1,
        wd: float = 0,
        momentum: float = 0,
        damping: float = 0,
        pooling_method: str = "mean",
        **kwargs,
    ):
        self.params_dict = {}
        for name, param in fast_weight.items():
            if param.requires_grad:
                self.params_dict[name] = param

        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.damping = damping
        self.use_momentum = momentum > 0
        self.pooling_fn = get_pooling_fn(pooling_method)

        self.setup_state()

    def step(
        self,
        lr_dict=None,
        wd_dict=None,
        momentum_dict=None,
        damping_dict=None,
    ):
        if lr_dict is None:
            lr_dict = {}
        if wd_dict is None:
            wd_dict = {}
        if momentum_dict is None:
            momentum_dict = {}
        if damping_dict is None:
            damping_dict = {}

        for name, p in self.params_dict.items():
            lr = self.pooling(lr_dict.get(name, self.lr))
            wd = self.pooling(wd_dict.get(name, self.wd))
            momentum = self.pooling(momentum_dict.get(name, self.momentum))
            damping = self.pooling(damping_dict.get(name, self.damping))
            lr = 1

            use_weight_decay = (not isinstance(wd, torch.Tensor)) and (wd != 0)
            use_momentum = (not isinstance(momentum, torch.Tensor)) and (momentum != 0)

            grad = p.grad
            if use_weight_decay:
                grad.add_(p.data * wd)

            if use_momentum:
                if self.state[name].get("momentum", None) is not None:
                    self.state[name]["momentum"] = torch.zeros_like(g)

                buf = self.state[name]["momentum"]
                buf.mul_(momentum).add_(grad * (1 - damping))
                grad = buf

            p.data.add_(-grad * lr)
