from .base_optimizer import FastWeightOptimizer
from .sgd import SGD


def get_optimizer(name: str, **kwargs) -> FastWeightOptimizer:
    if name == "sgd":
        return SGD(**kwargs)
    else:
        raise ValueError(f"Optimizer {name} not found")
