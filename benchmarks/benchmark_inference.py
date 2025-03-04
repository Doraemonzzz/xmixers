# adapt from https://github.com/fla-org/flash-linear-attention/blob/main/benchmarks/benchmark_generation.py

import argparse
import time
from typing import List, Optional

import torch
from torch.cuda import max_memory_allocated, memory_allocated
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM

import xmixers  # noqa


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def print_res(res: List[dict], columns: List[str]):
    print(",".join(columns))
    for r in res:
        print(",".join([str(r[c]).strip() for c in columns]))


def profile(
    cfg_path: str,
    name: str,
    vocab_size: int,
    batch_size_list: List[int] = [8],
    input_length_list: List[int] = [1],
    max_length_list: List[int] = [128],
    compile: bool = False,
    warmup_iter: int = 10,
    num_iter: int = 10,
    dtype: Optional[torch.dtype] = torch.bfloat16,
):
    device = torch.device("cuda")
    config = AutoConfig.from_pretrained(cfg_path)
    config.vocab_size = vocab_size
    model = AutoModelForCausalLM.from_config(config).cuda().to(dtype)
    num_parameters = model.num_parameters()
    embedding_parameters = (
        vocab_size * config.embed_dim * (1 if config.tie_word_embeddings else 2)
    )
    non_embedding_parameters = num_parameters - embedding_parameters
    print(f"Initializing {config.model_type} model from the config:\n{config}\n{model}")
    print(
        f"Number of parameters in total: {num_parameters} ({sizeof_fmt(num_parameters)})"
    )
    print(
        f"Number of non-embedding parameters in total: {non_embedding_parameters} ({sizeof_fmt(non_embedding_parameters)})"
    )
    print(
        f"Allocated memory after initialization: {sizeof_fmt(memory_allocated(device))}"
    )

    if args.compile:
        print("Compiling the model")
        model = torch.compile(model)
    model.eval()

    bar = trange(warmup_iter)

    columns = [
        "Model",
        "Params",
        "Params(non-embedding)",
        "Vocab_size",
        "Batch",
        "Sec",
        "Mem",
        "Warmup",
        "Input_length",
        "Max_length",
    ]
    res = []

    torch.cuda.synchronize()
    for batch_size, input_length, max_length in zip(
        batch_size_list, input_length_list, max_length_list
    ):
        input_ids = torch.randint(
            high=config.vocab_size, size=(batch_size, input_length), device=device
        )
        max_length = input_length + max_length

        print("Start warmup")
        for _ in bar:
            with torch.inference_mode():
                text = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    min_length=max_length,
                )
        print("End warmup")

        start = time.time()
        for _ in range(num_iter):
            with torch.inference_mode():
                text = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    min_length=max_length,
                )
        torch.cuda.synchronize()
        elapsed = time.time() - start

        res.append(
            {
                "Model": name,
                "Params": num_parameters / 1e6,
                "Params(non-embedding)": non_embedding_parameters / 1e6,
                "Vocab_size": vocab_size,
                "Batch": batch_size,
                "Sec": elapsed / num_iter * 1000,
                "Mem": sizeof_fmt(max_memory_allocated(device)),
                "Warmup": warmup_iter,
                "Input_length": input_length,
                "Max_length": max_length,
            }
        )

    print_res(res, columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation benchmarking")
    parser.add_argument("--cfg_path", default="llama_86m.json")
    parser.add_argument("--name", default="llama")
    parser.add_argument("--vocab_size", default=64000, type=int)
    parser.add_argument("--batch_size_list", nargs="+", type=int, default=[16])
    parser.add_argument("--input_length_list", nargs="+", type=int, default=[1])
    parser.add_argument("--max_length_list", nargs="+", type=int, default=[256])
    parser.add_argument("--warmup_iter", default=10, type=int)
    parser.add_argument("--num_iter", default=10, type=int)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    assert (
        len(args.batch_size_list)
        == len(args.input_length_list)
        == len(args.max_length_list)
    ), "Batch size list and input length list and max length list must have the same length"

    profile(
        cfg_path=args.cfg_path,
        name=args.name,
        vocab_size=args.vocab_size,
        batch_size_list=args.batch_size_list,
        input_length_list=args.input_length_list,
        max_length_list=args.max_length_list,
        compile=args.compile,
        warmup_iter=args.warmup_iter,
        num_iter=args.num_iter,
    )
