import logging
import math
import os
import sys

import torch.distributed as dist
from torch import nn

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("xmixers")


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def logging_info(string: str) -> None:
    if is_main_process():
        logger.info(string)


def print_params(**kwargs) -> None:
    if is_main_process():
        logger.info(f"start print config of {kwargs['__class__']}")
        for key in kwargs:
            if key in ["__class__", "self"]:
                continue
            logger.info(f"{key}: {kwargs[key]}")
        logger.info(f"end print config of {kwargs['__class__']}")


def print_config(config) -> None:
    if is_main_process():
        logger.info(f"start print config of {config['__class__']}")
        for key in config:
            if key in ["__class__", "self"]:
                continue
            logger.info(f"{key}: {config[key]}")
        logger.info(f"end print config of {config['__class__']}")


def print_module(module: nn.Module) -> str:
    named_modules_ = set()
    for p in module.named_modules():
        named_modules_.update([p[0]])
    named_modules = list(named_modules_)

    string_repr = ""
    for p in module.named_parameters():
        name = p[0].split(".")[0]
        if name not in named_modules:
            string_repr = (
                string_repr
                + "("
                + name
                + "): "
                + "Tensor("
                + str(tuple(p[1].shape))
                + ", requires_grad="
                + str(p[1].requires_grad)
                + ")\n"
            )

    return string_repr.rstrip("\n")


def next_power_of_2(n: int) -> int:
    return 2 ** (math.ceil(math.log(n, 2)))


def endswith(name, keyword_list):
    for keyword in keyword_list:
        if name.endswith(keyword):
            return True
    return False
