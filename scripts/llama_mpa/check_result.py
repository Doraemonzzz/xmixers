import argparse
import os

import torch
from metaseq.models.llama import LlamaLanguageModel
from transformers import AutoModelForCausalLM

import xmixers  # noqa

AUTO_DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}


def generate(model, x):
    b, n = x.shape
    y = []
    past_key_values = None
    for i in range(n):
        output = model(
            input_ids=x[:, i : i + 1],
            past_key_values=past_key_values,
        )
        past_key_values = output["past_key_values"]
        y.append(output["logits"])

    return torch.cat(y, dim=1)


def check_result(metaseq_dir, hf_dir, checkpoint_name, tokenizer_dir, dtype_name):
    metaseq_model_info = LlamaLanguageModel.from_pretrained(
        metaseq_dir,
        checkpoint_name,
        task={
            "vocab_filename": os.path.join(tokenizer_dir, "vocab.json"),
            "merges_filename": os.path.join(tokenizer_dir, "merges.txt"),
        },
    )

    dtype = AUTO_DTYPE_MAP[dtype_name]
    metaseq_model = metaseq_model_info["models"][0].cuda().to(dtype)
    print(metaseq_model)

    hf_model = AutoModelForCausalLM.from_pretrained(hf_dir).cuda().to(dtype)
    print(hf_model)

    layers = len(hf_model.model.layers)
    print(f"layers: {layers}")

    print("Check embedding and out proj")
    print(
        torch.norm(
            metaseq_model.decoder.embed_tokens.weight
            - hf_model.model.embed_tokens.weight
        ),
    )
    print(
        torch.norm(
            metaseq_model.decoder.output_projection.weight - hf_model.lm_head.weight
        ),
    )

    print("Check average")
    print(
        torch.mean(metaseq_model.decoder.embed_tokens.weight),
        torch.mean(hf_model.model.embed_tokens.weight),
    )
    print(
        torch.mean(metaseq_model.decoder.output_projection.weight),
        torch.mean(hf_model.lm_head.weight),
    )
    print("=====Start checking weight diff=====")
    print(
        torch.norm(
            metaseq_model.decoder.embed_tokens.weight
            - hf_model.model.embed_tokens.weight
        )
    )
    print(
        torch.norm(
            metaseq_model.decoder.output_projection.weight - hf_model.lm_head.weight
        )
    )
    for i in range(layers):
        print(f"layer {i}")
        ##### token mixer
        print("qkv")
        # qkv
        # q
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].token_mixer.q_proj.weight
                - hf_model.model.layers[i].token_mixer.q_proj.weight
            )
        )
        # kv
        d = hf_model.model.layers[i].token_mixer.k_proj.weight.shape[0]
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].token_mixer.kv2_proj.weight[:d]
                - hf_model.model.layers[i].token_mixer.k_proj.weight
            )
        )
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].token_mixer.kv2_proj.weight[d:]
                - hf_model.model.layers[i].token_mixer.v_proj.weight
            )
        )
        # kv head
        if hasattr(hf_model.model.layers[i].token_mixer, "k_head_proj"):
            d = (
                metaseq_model.decoder.layers[i].token_mixer.kv1_proj.weight.shape[0]
                // 2
            )
            print(
                torch.norm(
                    metaseq_model.decoder.layers[i].token_mixer.kv1_proj.weight[:d]
                    - hf_model.model.layers[i].token_mixer.k_head_proj.weight
                )
            )
            print(
                torch.norm(
                    metaseq_model.decoder.layers[i].token_mixer.kv1_proj.weight[d:]
                    - hf_model.model.layers[i].token_mixer.v_head_proj.weight
                )
            )
        else:
            d = metaseq_model.decoder.layers[i].token_mixer.kv1_proj.shape[0] // 2
            print(
                torch.norm(
                    metaseq_model.decoder.layers[i].token_mixer.kv1_proj[:d]
                    - hf_model.model.layers[i].token_mixer.k_head
                )
            )
            print(
                torch.norm(
                    metaseq_model.decoder.layers[i].token_mixer.kv1_proj[d:]
                    - hf_model.model.layers[i].token_mixer.v_head
                )
            )
        print("o proj")
        # o proj
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].token_mixer.out_proj.weight
                - hf_model.model.layers[i].token_mixer.out_proj.weight
            )
        )

        ##### channel mixer
        print("glu")
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].channel_mixer.l1.weight
                - hf_model.model.layers[i].channel_mixer.w1.weight
            )
        )
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].channel_mixer.l2.weight
                - hf_model.model.layers[i].channel_mixer.w2.weight
            )
        )
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].channel_mixer.l3.weight
                - hf_model.model.layers[i].channel_mixer.w3.weight
            )
        )
    print("=====End checking weight diff=====")

    b = 2
    m = 50272

    # train test
    for n in [128]:
        input = torch.randint(0, m, (b, n)).cuda()
        o1 = metaseq_model(input)[0]
        hf_model.train()
        o2 = hf_model(input)["logits"]
        hf_model.eval()
        print("generate")
        o3 = generate(hf_model, input)

        print(f"n: {n}")
        print("training diff")
        print(torch.norm(o1 - o2))
        print("inference diff")
        print(torch.norm(o1 - o3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--metaseq_dir",
        type=str,
    )
    parser.add_argument(
        "--hf_dir",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
    )
    parser.add_argument("--dtype_name", type=str)
    args = parser.parse_args()
    check_result(
        args.metaseq_dir,
        args.hf_dir,
        args.checkpoint_name,
        args.tokenizer_dir,
        args.dtype_name,
    )
