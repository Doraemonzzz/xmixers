# adapt from https://github.com/sustcsonglin/flash-linear-attention/blob/main/benchmarks/benchmark_training_throughput.py

import argparse
import time
from typing import List, Optional, Tuple

import torch
from accelerate import Accelerator
from torch.cuda import max_memory_allocated, memory_allocated
from torch.optim import AdamW
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.optimization import get_cosine_schedule_with_warmup

import xmixers  # noqa


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:.2f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f}Yi{suffix}"


def prepare_inputs(
    batch_size: int, seq_len: int, varlen: bool, vocab_size: int, device: torch.device
):
    if varlen:
        tokens = torch.randint(
            high=vocab_size, size=(1, batch_size * seq_len), device=device
        )
        offsets = torch.cat(
            [
                torch.tensor([0], dtype=torch.long, device=device),
                torch.randperm(batch_size * seq_len - 16, device=device)[
                    : batch_size - 1
                ]
                + 16,
                torch.tensor([batch_size * seq_len], dtype=torch.long, device=device),
            ],
            0,
        ).sort()[0]
    else:
        tokens = torch.randint(
            high=vocab_size, size=(batch_size, seq_len), device=device
        )
        offsets = None
    return tokens, offsets


def profile(
    cfg_path: str,
    name: str,
    vocab_size: int = 64000,
    batch_size_list: List[int] = [8],
    seq_len_list: List[int] = [2048],
    varlen: bool = False,
    warmup_steps: int = 16,
    steps: int = 32,
    total_steps: int = 1024,
    lr: float = 3e-4,
    betas: Tuple[float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    mixed_precision: str = "bf16",
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

    accelerator = Accelerator(mixed_precision=mixed_precision)
    optimizer = AdamW(
        model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, fused=True
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)

    bar = trange(warmup_steps)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    for seq_len, batch_size in zip(seq_len_list, batch_size_list):
        torch.cuda.synchronize(device)
        for _ in bar:
            # forward pass
            tokens, offsets = prepare_inputs(
                batch_size=batch_size,
                seq_len=seq_len,
                varlen=varlen,
                vocab_size=config.vocab_size,
                device=device,
            )
            outputs = model(tokens, labels=tokens, offsets=offsets)
            # backward pass
            accelerator.backward(outputs.loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            bar.set_description_str(
                f"Max memory allocated: {sizeof_fmt(max_memory_allocated(device))}"
            )

        start, total_tokens = time.time(), 0
        bar = trange(steps)
        torch.cuda.synchronize(device)
        for _ in bar:
            # forward pass
            tokens, offsets = prepare_inputs(
                batch_size=batch_size,
                seq_len=seq_len,
                varlen=varlen,
                vocab_size=config.vocab_size,
                device=device,
            )
            outputs = model(tokens, labels=tokens, offsets=offsets)
            # backward pass
            accelerator.backward(outputs.loss)
            optimizer.step()
            optimizer.zero_grad()

            total_tokens += batch_size * seq_len
            torch.cuda.synchronize(device)
            duration = time.time() - start
            bar.set_description_str(
                f"Thoughput: {total_tokens / duration:10.2f} tokens/s"
            )

        print(
            f"Model,Params,Params(non-embedding),Tgs,Mem,Batch,Seq,Varlen,Warmup,Steps"
        )
        print(
            f"{name},{num_parameters / 1e6:10.2f},{non_embedding_parameters / 1e6:10.2f},{total_tokens / duration:10.2f},{sizeof_fmt(max_memory_allocated(device))},{batch_size},{seq_len},{varlen},{warmup_steps},{steps}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", default="llama_86m.json")
    parser.add_argument("--name", default="llama")
    parser.add_argument("--vocab_size", default=64000, type=int)
    parser.add_argument("--batch_size_list", nargs="+", type=int, default=[8])
    parser.add_argument("--seq_len_list", nargs="+", type=int, default=[2048])
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--warmup_steps", default=16, type=int)
    parser.add_argument("--steps", default=32, type=int)
    args = parser.parse_args()

    profile(
        cfg_path=args.cfg_path,
        name=args.name,
        vocab_size=args.vocab_size,
        batch_size_list=args.batch_size_list,
        seq_len_list=args.seq_len_list,
        varlen=args.varlen,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
    )
