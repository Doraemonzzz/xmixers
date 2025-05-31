import torch


def get_pooling_fn(pooling_method: str):
    if pooling_method == "mean":

        def f(x):
            return torch.mean(x, dim=1).unsqueeze(-1).unsqueeze(-1)

    else:
        raise ValueError(f"Pooling method {pooling_method} not found")

    return f
