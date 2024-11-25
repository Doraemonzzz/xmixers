# coding=utf-8
"""Convert Hgrn checkpoint."""


import argparse
from pathlib import Path

import torch
from transformers.utils import logging

from xmixers.models import Hgrn3Config, Hgrn3ForCausalLM

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_checkpoint(checkpoint_path, tokenizer_path, vocab_size=-1):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    # config_dict = {"vocab_size": 64000}
    config_dict = {
        "vocab_size": vocab_size,
        "q_activation": "silu",
        "k_activation": "silu",
        "beta_activation": "silu",
    }
    print("======Get Config Start======")
    for key in sd["cfg"]["model"]:
        print(f'{key}={sd["cfg"]["model"][key]},')
        if key == "share_decoder_input_output_embed":
            config_dict["tie_word_embeddings"] = sd["cfg"]["model"][key]
    # print()
    config_keys = {
        "decoder_embed_dim",
        "expand_ratio",
        "decoder_layers",
        "add_bos_token",
        "causal",
        "glu_dim",
        "mid_dim",
        "bias",
        "use_norm",
        "norm_type",
        "no_scale_embedding",
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
            if key in ["glu_dim", "decoder_layers"]:
                if key == "glu_dim":
                    config_dict["mid_dim"] = int(sd["cfg"]["model"][key])
                    print(f'mid_dim = {config_dict["mid_dim"]}')
                elif key == "decoder_layers":
                    config_dict["num_layers"] = int(sd["cfg"]["model"][key])
                    print(f'num_layers = {config_dict["num_layers"]}')

            else:
                if key == "expand_ratio":
                    config_dict[key] = int(sd["cfg"]["model"][key])
                elif key in ["q_activation", "k_activation", "beta_activation"]:
                    config_dict[key] = "silu"
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
        if "hgru.lambda_" in key and "lambda_proj" not in key:
            continue
        if "lower_bound" in key:
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
        new_key = key.replace("token_mixer.hgru.", "token_mixer.")
        new_key = new_key.replace("decoder.", "model.")
        # token mixer
        new_key = new_key.replace("lambda_proj", "f_proj")
        new_key = new_key.replace("output_gate", "out_gate")
        # channel mixer
        new_key = new_key.replace("l1.weight", "w1.weight")
        new_key = new_key.replace("l2.weight", "w2.weight")
        new_key = new_key.replace("l3.weight", "w3.weight")

        if "in_proj" in new_key:
            d = value.shape[1]
            for i, name in enumerate(["q_proj", "k_proj", "v_proj"]):
                state_dict[new_key.replace("in_proj", name)] = value[
                    i * d : (i + 1) * d
                ]
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


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path,
    tokenizer_path,
    pytorch_dump_folder_path,
    config=None,
    vocab_size=-1,
):
    state_dict, config_dict = load_checkpoint(
        checkpoint_path, tokenizer_path, vocab_size
    )
    config = Hgrn3Config(**config_dict)
    model = Hgrn3ForCausalLM(config)
    print(model)
    res = model.load_state_dict(state_dict)
    print(res)
    print(model)
    # Check results
    print(torch.mean(model.model.embed_tokens.weight), torch.mean(model.lm_head.weight))
    Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


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
    args = parser.parse_args()
    convert_checkpoint(
        args.fairseq_path,
        args.tokenizer_path,
        args.pytorch_dump_folder_path,
        config=args.hf_config,
        vocab_size=args.vocab_size,
    )
