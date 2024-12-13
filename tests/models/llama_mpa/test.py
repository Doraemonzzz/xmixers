import torch
from transformers import AutoModelForCausalLM

import xmixers  # noqa
from xmixers import LLaMAConfig

AUTO_DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}


def generate(model, x):
    model.eval()
    b, n = x.shape
    y = []
    past_key_values = None

    with torch.inference_mode():
        for i in range(0, n):
            output = model(
                input_ids=x[:, i : i + 1],
                past_key_values=past_key_values,
            )
            past_key_values = output["past_key_values"]
            y.append(output["logits"])

    y = torch.cat(y, dim=1)
    return y


def main(dtype_name="bf16"):
    dtype = AUTO_DTYPE_MAP[dtype_name]
    # config = LLaMAConfig(norm_type="srmsnorm")
    config = LLaMAConfig()
    hf_model = AutoModelForCausalLM.from_config(config).cuda().to(dtype)
    print(config)
    print(hf_model)
    hf_model.eval()

    b = 2
    m = 50272

    for n in [32, 64, 128, 256, 512, 1024]:
        input = torch.randint(0, m, (b, n)).cuda()
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            with torch.inference_mode():
                o1 = hf_model(input)["logits"]

            o2 = generate(hf_model, input)
        print(f"n: {n}, diff: {torch.norm(o1 - o2)}")


if __name__ == "__main__":
    dtype_name = "bf16"
    dtype_name = "fp32"
    # seed = 42 # any number
    # set_deterministic(seed=seed)
    main(dtype_name=dtype_name)
