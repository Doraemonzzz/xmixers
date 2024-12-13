# coding=utf-8
"""Convert llama_mpa checkpoint."""


import argparse
import json
import os
from pathlib import Path
from shutil import copy2

import torch
from transformers.utils import logging

from xmixers.models import LLaMAConfig, LLaMAForCausalLM

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

AUTO_DTYPE_MAP = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def load_checkpoint(checkpoint_path, tokenizer_path, vocab_size=-1):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")

    config_dict = {
        "vocab_size": vocab_size,
        "token_mixer_type": "mpa",
    }
    print("======Get Config Start======")
    for key in sd["cfg"]["model"]:
        print(f'{key}={sd["cfg"]["model"][key]},')
        if key == "share_decoder_input_output_embed":
            config_dict["tie_word_embeddings"] = sd["cfg"]["model"][key]

    config_keys = {
        "decoder_embed_dim",
        "num_heads",
        "decoder_layers",
        "add_bos_token",
        "causal",
        "glu_dim",
        "mid_dim",
        "bias",
        "use_norm",
        "norm_type",
        "no_scale_embedding",
        "glu_act",
        "use_mpa",
        "llama_core_matrix",
        "max_target_positions",
        "head_dim",
    }
    print("======Model Config======")
    for key in sd["cfg"]["model"]:
        if key in config_keys:
            print(f"self.{key} = {key}")
    print("======Get Config End======")

    # config dict
    print("======Get Model Config Start======")
    for key in sd["cfg"]["model"]:
        if key in config_keys:
            if key in [
                "glu_dim",
                "decoder_layers",
                "use_embed_scale",
                "decoder_embed_dim",
                "glu_act",
                "use_mpa",
                "llama_core_matrix",
                "no_scale_embedding",
                "max_target_positions",
            ]:
                if key == "glu_dim":
                    config_dict["mid_dim"] = int(sd["cfg"]["model"][key])
                    print(f'mid_dim = {config_dict["mid_dim"]}')
                elif key == "decoder_layers":
                    config_dict["num_layers"] = int(sd["cfg"]["model"][key])
                    print(f'num_layers = {config_dict["num_layers"]}')
                elif key == "no_scale_embedding":
                    config_dict["use_embed_scale"] = not sd["cfg"]["model"][key]
                elif key == "decoder_embed_dim":
                    config_dict["embed_dim"] = int(sd["cfg"]["model"][key])
                elif key == "glu_act":
                    config_dict["glu_activation"] = sd["cfg"]["model"][key]
                elif key == "use_mpa":
                    mpa = sd["cfg"]["model"][key]
                    if mpa in [2, 3]:
                        config_dict["mpa_type"] = 0
                    else:
                        config_dict["mpa_type"] = 1
                    if mpa in [2, 4]:
                        config_dict["mpa_activation"] = "none"
                    else:
                        config_dict["mpa_activation"] = "sigmoid"
                elif key == "llama_core_matrix":
                    llama_core_matrix = int(sd["cfg"]["model"][key])
                    if llama_core_matrix == 12:
                        config_dict["lrpe_type"] = 1
                    elif llama_core_matrix == 4:
                        config_dict["lrpe_type"] = 2
                    elif llama_core_matrix == 13:
                        config_dict["lrpe_type"] = 6
                    elif llama_core_matrix == 16:
                        config_dict["lrpe_type"] = 5
                elif key == "max_target_positions":
                    config_dict["max_position_embeddings"] = int(
                        sd["cfg"]["model"][key]
                    )
            else:
                if key == "num_heads":
                    # config_dict[key] = int(
                    #     sd["cfg"]["model"]["decoder_embed_dim"] // 128
                    # )
                    config_dict[key] = int(
                        sd["cfg"]["model"]["decoder_attention_heads"]
                    )
                elif key == "norm_type":
                    config_dict[key] = sd["cfg"]["model"][key].replace(
                        "simplermsnorm", "srmsnorm"
                    )
                else:
                    config_dict[key] = sd["cfg"]["model"][key]
                print(f"{key} = {config_dict[key]}")
    print("======Get Model Config End======")

    # model state dict
    origin_state_dict = sd["model"]
    state_dict = {}
    keys = list(origin_state_dict.keys())

    print("======Origin======")
    for key in origin_state_dict:
        print(key)

    for key in keys:
        value = origin_state_dict[key]
        if "theta" in key:
            continue
        if key == "decoder.version":
            continue
        if key == "decoder.output_projection":
            new_key = "lm_head.weight"
            state_dict[new_key] = value
            continue
        if key == "decoder.output_projection.weight":
            new_key = "lm_head.weight"
            print(torch.mean(value))
            state_dict[new_key] = value
            continue
        new_key = key.replace("decoder.", "model.")
        # channel mixer
        new_key = new_key.replace("l1.weight", "w1.weight")
        new_key = new_key.replace("l2.weight", "w2.weight")
        new_key = new_key.replace("l3.weight", "w3.weight")

        if "kv2_proj" in new_key:
            state_dict[new_key.replace("kv2_proj", "kv_proj")] = value

            # old version
            # d = value.shape[0] // 2
            # for i, name in enumerate(["k_proj", "v_proj"]):
            #     state_dict[new_key.replace("kv2_proj", name)] = value[
            #         i * d : (i + 1) * d
            #     ]
        elif "kv1_proj" in new_key:
            if len(value.shape) == 2:
                state_dict[new_key.replace("kv1_proj", "kv_head_proj")] = value
            else:
                state_dict[new_key.replace("kv1_proj", "kv_head")] = value

            # d = value.shape[0] // 2
            # if len(value.shape) == 2:
            #     keys = ["k_head_proj", "v_head_proj"]
            # else:
            #     keys = ["k_head", "v_head"]

            # for i, name in enumerate(keys):
            #     state_dict[new_key.replace("kv1_proj", name)] = value[
            #         i * d : (i + 1) * d
            #     ]
        else:
            state_dict[new_key] = value

    if "decoder.output_projection.weight" not in keys:
        if "model.embed_tokens.weight" in state_dict:
            # for 3b
            state_dict["lm_head.weight"] = state_dict[
                "model.embed_tokens.weight"
            ].clone()
        else:
            state_dict["lm_head.weight"] = (
                state_dict["decoder.embed_tokens.weight"].transpose(1, 0).clone()
            )

    print("======Transform======")
    for key in state_dict:
        print(key, state_dict[key].shape, torch.mean(state_dict[key]))

    for key in config_dict:
        print(key, config_dict[key])

    return state_dict, config_dict


def get_json_key(path, key):
    with open(path) as f:
        data = json.load(f)

    if key in data.keys():
        return data[key]
    else:
        return None


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path,
    tokenizer_path,
    pytorch_dump_folder_path,
    config=None,
    vocab_size=-1,
    tie_weights=False,
    dtype="bf16",
    max_shard_size="5GB",
):
    state_dict, config_dict = load_checkpoint(
        checkpoint_path, tokenizer_path, vocab_size
    )
    config = LLaMAConfig(**config_dict)
    model = LLaMAForCausalLM(config)
    if tie_weights:
        model.tie_weights()  # ·Explicitly tie weights·if·necessary
    print(model)
    res = model.load_state_dict(state_dict)
    print(res)
    print(model)
    # Check results
    print(torch.mean(model.model.embed_tokens.weight), torch.mean(model.lm_head.weight))
    model.to(AUTO_DTYPE_MAP[dtype])
    Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        pytorch_dump_folder_path, max_shard_size=max_shard_size, safe_serialization=True
    )


def check_max(path):
    """Ensure model_max_length in tokenizer config matches max_position_embeddings in model config"""
    config_path = os.path.join(path, "config.json")
    tokenizer_path = os.path.join(path, "tokenizer_config.json")

    config_max = get_json_key(config_path, "max_position_embeddings")
    tokenizer_max = get_json_key(tokenizer_path, "model_max_length")

    if config_max != tokenizer_max:
        with open(tokenizer_path) as f:
            tokenizer_config = json.load(f)

        old_max = tokenizer_config["model_max_length"]
        tokenizer_config["model_max_length"] = config_max

        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=4)
            f.write("\n")

        print(f"Updated {tokenizer_path}: model_max_length {old_max} -> {config_max}")


def update_codebase(path):
    """Copy tokenizer files and sync configs"""

    for file in os.listdir("./tokenizer"):
        copy2(src=os.path.join("./tokenizer", file), dst=os.path.join(path, file))

    # Ensure configs are in sync
    check_max(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fairseq_path",
        type=str,
        help=("path to fairseq checkpoint in correct format."),
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help=("path to tokenizer path."),
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument("--hf_config", default=None, type=str, help="Define HF config.")
    parser.add_argument("--tie_weights", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
    )
    parser.add_argument("--max_shard_size", type=str, default="5GB")

    args = parser.parse_args()
    convert_checkpoint(
        args.fairseq_path,
        args.tokenizer_path,
        args.pytorch_dump_folder_path,
        config=args.hf_config,
        vocab_size=args.vocab_size,
        tie_weights=args.tie_weights,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
    )
    try:
        update_codebase(args.pytorch_dump_folder_path)
    except:
        pass
