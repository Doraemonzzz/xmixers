import argparse
import os

import torch
from metaseq.models.hgrn import HgrnLanguageModel
from transformers import AutoModelForCausalLM


def check_result(metaseq_dir, hf_dir, checkpoint_name, tokenizer_dir):
    metaseq_model_info = HgrnLanguageModel.from_pretrained(
        metaseq_dir,
        checkpoint_name,
        task={
            "vocab_filename": os.path.join(tokenizer_dir, "vocab.json"),
            "merges_filename": os.path.join(tokenizer_dir, "merges.txt"),
        },
    )

    dtype = torch.bfloat16
    # dtype = torch.float32
    metaseq_model = metaseq_model_info["models"][0].cuda().to(dtype)
    layers = 6
    print(f"layers: {layers}")
    print(metaseq_model)

    hf_model = AutoModelForCausalLM.from_pretrained(hf_dir).cuda().to(dtype)
    print(hf_model)

    print("Check average")
    print(
        torch.mean(metaseq_model.decoder.embed_tokens.weight),
        torch.mean(hf_model.model.embed_tokens.weight),
    )
    print(
        torch.mean(metaseq_model.decoder.output_projection.weight),
        torch.mean(hf_model.lm_head.weight),
    )
    print("=====Start checking lower bound=====")
    # print(dir(metaseq_model.decoder))
    # print(dir(hf_model.model))
    print(metaseq_model.decoder.lower_bound)
    print(hf_model.model.log_lower_bound)
    print(
        torch.norm(metaseq_model.decoder.lower_bound - hf_model.model.log_lower_bound)
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
    print(
        torch.norm(metaseq_model.decoder.lower_bound - hf_model.model.log_lower_bound)
    )
    for i in range(layers):
        ##### token mixer
        print("qkv")
        # qkv
        d = metaseq_model.decoder.layers[i].token_mixer.hgru.in_proj.weight.shape[1]
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].token_mixer.hgru.in_proj.weight[:d]
                - hf_model.model.layers[i].token_mixer.q_proj.weight
            )
        )
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].token_mixer.hgru.in_proj.weight[
                    d : 2 * d
                ]
                - hf_model.model.layers[i].token_mixer.k_proj.weight
            )
        )
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].token_mixer.hgru.in_proj.weight[2 * d :]
                - hf_model.model.layers[i].token_mixer.v_proj.weight
            )
        )
        print("o proj")
        # o proj
        print(
            torch.norm(
                metaseq_model.decoder.layers[i].token_mixer.hgru.out_proj.weight
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
    hf_model.train()

    # for n in [1, 2, 63, 127]:
    # for n in [1, 5, 123, 1024]:
    # for n in [512, 1024]:
    # for n in [1, 5, 126]:
    # for n in [5]:
    for n in [128]:
        input = torch.randint(0, m, (b, n)).cuda()
        o1 = metaseq_model(input)[0]
        # hf_model.eval()
        o2 = hf_model(input)["logits"]
        print(o1.shape, o2.shape)
        print(f"n: {n}")
        print(torch.norm(o1 - o2))
        # print(o1.shape)
        # print(o1[0, -1, -5:])
        # print(o2[0, -1, -5:])

    # hf_model.eval()
    # input = torch.randint(0, m, (b, n)).cuda()
    # print("generate")
    # generated = hf_model.generate(
    #     input,
    #     max_length=5,
    # )
    # print(generated.shape)


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
    args = parser.parse_args()
    check_result(
        args.metaseq_dir, args.hf_dir, args.checkpoint_name, args.tokenizer_dir
    )
