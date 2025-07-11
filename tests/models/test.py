from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import xmixers  # noqa
from xmixers.models import (
    DecayLinearTransformerConfig,
    DeltaNetConfig,
    DenseRnnConfig,
    FlexGPTConfig,
    GsaConfig,
    Hgrn1Config,
    Hgrn2Config,
    Hgrn3Config,
    ImplicitLinearTransformerConfig,
    LightNetConfig,
    LinearTransformerConfig,
    LLaMAConfig,
    Mamba2XmixersConfig,
    MesaNetConfig,
    TnlConfig,
    TTTConfig,
)

AUTO_DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}


def get_config(model_type):
    if model_type == "llama":
        config = LLaMAConfig()
    elif model_type == "mpa":
        config = LLaMAConfig()
        config.token_mixer_type = "mpa"
        config.lrpe_type = 3
    elif model_type == "tpa":
        config = LLaMAConfig()
        config.token_mixer_type = "tpa"
        config.lrpe_type = 3
        config.q_rank = 8
        config.kv_rank = 2
    elif model_type == "mla":
        config = LLaMAConfig()
        config.token_mixer_type = "mla"
        config.lrpe_type = 3
    elif model_type == "hgrn2":
        config = Hgrn2Config()
    elif model_type == "hgrn2_scalar_decay":
        config = Hgrn2Config()
        config.token_mixer_type = "hgru2_scalar_decay"
    elif model_type == "lightnet":
        config = LightNetConfig()
    elif model_type == "lightnet_no_tpe":
        config = LightNetConfig()
        config.use_tpe = False
    elif model_type == "lightnet_scalar_decay":
        config = LightNetConfig()
        config.scalar_decay = True
    elif model_type == "lightnet_no_tpe_scalar_decay":
        config = LightNetConfig()
        config.use_tpe = False
        config.scalar_decay = True
    elif model_type == "tnl":
        config = TnlConfig()
    elif model_type == "tnl_state":
        config = TnlConfig()
        config.use_initial_state = True
    elif model_type == "linear_transformer":
        config = LinearTransformerConfig()
    elif model_type == "linear_transformer_no_tpe":
        config = LinearTransformerConfig()
        config.use_tpe = False
    elif model_type == "cosformer2":
        config = LinearTransformerConfig()
        config.lrpe_type = 6
    elif model_type == "cosformer2_no_tpe":
        config = LinearTransformerConfig()
        config.use_tpe = False
        config.lrpe_type = 6
    elif model_type == "naive_deltanet":
        config = DeltaNetConfig()
        config.use_decay = False
        config.scalar_decay = False
    elif model_type == "scalar_decay_deltanet":
        config = DeltaNetConfig()
        config.use_decay = True
        config.scalar_decay = True
    elif model_type == "scalar_decay_lower_bound_deltanet":
        config = DeltaNetConfig()
        config.use_decay = True
        config.scalar_decay = True
        config.use_lower_bound = True
    elif model_type == "vector_decay_deltanet":
        config = DeltaNetConfig()
        config.use_decay = True
        config.scalar_decay = False
    elif model_type == "vector_decay_lower_bound_deltanet":
        config = DeltaNetConfig()
        config.use_decay = True
        config.scalar_decay = False
        config.use_lower_bound = True
    elif model_type == "dense_rnn":
        config = DenseRnnConfig()
    elif model_type == "dense_rnn_lower_bound":
        config = DenseRnnConfig()
        config.use_lower_bound = True
    elif model_type == "decay_linear_transformer_hgrn2":
        config = DecayLinearTransformerConfig()
        config.use_lower_bound = True
        config.share_decay = True
    elif model_type == "decay_linear_transformer_hgrn2_rope":
        config = DecayLinearTransformerConfig()
        config.use_lower_bound = True
        config.share_decay = True
        config.use_lrpe = True
    elif model_type == "decay_linear_transformer_hgrn2_scalar_decay":
        config = DecayLinearTransformerConfig()
        config.use_lower_bound = True
        config.scalar_decay = True
        config.share_decay = False
    elif model_type == "decay_linear_transformer_hgrn2_rope_scalar_decay":
        config = DecayLinearTransformerConfig()
        config.use_lower_bound = True
        config.scalar_decay = True
        config.share_decay = False
        config.use_lrpe = True
    elif model_type == "decay_linear_transformer_mamba":
        config = DecayLinearTransformerConfig()
        config.decay_type = "mamba"
        config.share_decay = False
        config.use_tpe = False
    elif model_type == "decay_linear_transformer_mamba_scalar_decay":
        config = DecayLinearTransformerConfig()
        config.decay_type = "mamba"
        config.scalar_decay = True
        config.share_decay = False
        config.use_tpe = False
    elif model_type == "decay_linear_transformer_gla":
        config = DecayLinearTransformerConfig()
        config.decay_type = "gla"
        config.share_decay = False
    elif model_type == "decay_linear_transformer_gla_scalar_decay":
        config = DecayLinearTransformerConfig()
        config.decay_type = "gla"
        config.scalar_decay = True
        config.share_decay = False
    elif model_type == "decay_linear_transformer_lightnet":
        config = DecayLinearTransformerConfig()
        config.decay_type = "lightnet"
        config.share_decay = False
    elif model_type == "decay_linear_transformer_lightnet_share_decay":
        config = DecayLinearTransformerConfig()
        config.decay_type = "lightnet"
        config.share_decay = True
        config.scalar_decay = False
    elif model_type == "decay_linear_transformer_lightnet_scalar_decay":
        config = DecayLinearTransformerConfig()
        config.decay_type = "lightnet"
        config.scalar_decay = True
    elif model_type == "decay_linear_transformer_lssp":
        config = DecayLinearTransformerConfig()
        config.decay_type = "lssp"
        config.share_decay = False
    elif model_type == "decay_linear_transformer_lssp_scalar_decay":
        config = DecayLinearTransformerConfig()
        config.decay_type = "lssp"
        config.scalar_decay = True
    elif model_type == "decay_linear_transformer_tnl":
        config = DecayLinearTransformerConfig()
        config.decay_type = "tnl"
        config.scalar_decay = True
    elif model_type == "decay_linear_transformer_tnl_scalar_decay":
        config = DecayLinearTransformerConfig()
        config.decay_type = "tnl"
        config.scalar_decay = True
    elif model_type == "nsa":
        config = LLaMAConfig()
        config.token_mixer_type = "nsa"
        config.head_dim = 64
        config.num_heads = 16
        config.kv_heads = 1
        config.window_size = 128
    elif model_type == "gsa":
        config = GsaConfig()
    elif model_type == "ttt":
        config = TTTConfig()
    elif model_type == "hgrn3":
        config = Hgrn3Config()
    elif model_type == "hgrn3_scalar_decay":
        config = Hgrn3Config()
        config.scalar_decay = True
    elif model_type == "alibi":
        config = FlexGPTConfig()
        config.rpe_type = 1
    elif model_type == "fox":
        config = FlexGPTConfig()
        config.token_mixer_type = "forgetting_attn"
    elif model_type == "fox_window":
        config = FlexGPTConfig()
        config.token_mixer_type = "forgetting_attn"
        config.window_size = 4
    elif model_type == "sb_attn":
        config = FlexGPTConfig()
        config.token_mixer_type = "sb_attn"
    elif model_type == "mfa":
        config = LLaMAConfig()
        config.token_mixer_type = "mfa"
    elif model_type == "mfa_kv_share":
        config = LLaMAConfig()
        config.token_mixer_type = "mfa"
        config.share_kv = True
    elif model_type == "hgrn1":
        config = Hgrn1Config()
    elif model_type == "mamba2":
        config = Mamba2XmixersConfig()
    elif model_type == "scalar_decay_delta_product_net":
        config = DeltaNetConfig()
        config.token_mixer_type = "delta_product_unit"
        config.rank = 2
        config.use_decay = True
        config.scalar_decay = True
    elif model_type == "implicit_value_attn":
        config = ImplicitLinearTransformerConfig()
        config.token_mixer_type = "implicit_value_attn"
    elif model_type == "mesa_net":
        config = MesaNetConfig()
    elif model_type == "poly_net":
        config = LLaMAConfig()
        config.token_mixer_type = "poly_attn"
    elif model_type == "kernel_regression_attn":
        config = LLaMAConfig()
        config.token_mixer_type = "kernel_regression_attn"
        config.use_decay = True
        config.threshold = 0.99
    elif model_type == "kernel_regression_attn_no_decay":
        config = LLaMAConfig()
        config.token_mixer_type = "kernel_regression_attn"
        config.use_decay = False
        config.threshold = 0.99
    elif model_type == "kernel_regression_attn_no_kr":
        config = LLaMAConfig()
        config.token_mixer_type = "kernel_regression_attn"
        config.use_kernel_regression = False
        config.scale_type = 1
    elif model_type == "path_attn":
        config = LLaMAConfig()
        config.token_mixer_type = "path_attn"
        config.use_decay = True
        config.threshold = 0.99
        config.num_heads = 16
        config.embed_dim = 1024
    elif model_type == "path_attn_no_decay":
        config = LLaMAConfig()
        config.token_mixer_type = "path_attn"
        config.use_decay = False
        config.threshold = 0.99
        config.num_heads = 16
        config.embed_dim = 1024
    elif model_type == "path_attn_no_beta":
        config = LLaMAConfig()
        config.token_mixer_type = "path_attn"
        config.use_beta = False
        config.threshold = 0.99
        config.num_heads = 16
        config.embed_dim = 1024
    elif model_type == "fox_offset":
        config = FlexGPTConfig()
        config.token_mixer_type = "forgetting_attn"
        config.use_offset = True
        config.threshold = 0.99
    elif model_type == "kernel_regression_attn_qk_norm":
        config = LLaMAConfig()
        config.token_mixer_type = "kernel_regression_attn"
        config.use_qk_norm = True

    return config


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


def main(args):
    dtype_name = args.dtype
    model_type = args.model_type

    device = torch.device("cuda")
    dtype = AUTO_DTYPE_MAP[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    m = len(tokenizer)
    n = 32

    config = get_config(model_type)
    config.vocab_size = len(tokenizer)
    config.num_layers = 2

    hf_model = AutoModelForCausalLM.from_config(config).to(device).to(dtype)
    hf_model.post_init_weights()
    print(config)
    print(hf_model)
    hf_model.eval()

    print("-" * 5, "Start test generate without attention mask", "-" * 5)

    for b in [1, 2]:
        for n in [32]:
            input = torch.randint(0, m, (b, n)).to(device)
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                with torch.inference_mode():
                    o1 = hf_model(input)["logits"]

                o2 = generate(hf_model, input)
            print(f"b: {b}, n: {n}, diff: {torch.norm(o1 - o2)}")

    print("-" * 5, "End test generate without attention mask", "-" * 5)

    print("-" * 5, "Start test generate with attention mask", "-" * 5)
    texts = [
        "The weather today is",
        "Artificial Intelligence is rapidly",
        "I've been learning to code",
        "I'm going to the store",
    ]
    inputs = tokenizer(
        texts, padding=True, return_tensors="pt", padding_side="left"
    ).to(device)

    # batch compute
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        with torch.inference_mode():
            o1 = hf_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )["logits"][:, -1]

    # single compute
    o2 = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            with torch.inference_mode():
                o2.append(
                    hf_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )["logits"][:, -1]
                )
    o2 = torch.cat(o2, dim=0)
    print(f"n: {n}, diff: {torch.norm(o1 - o2)}")

    print("-" * 5, "End test generate with attention mask", "-" * 5)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument(
        "--model_type",
        type=str,
        default="llama",
        choices=[
            "llama",
            "mpa",
            "tpa",
            "hgrn2",
            "hgrn2_scalar_decay",
            "lightnet",
            "lightnet_no_tpe",
            "lightnet_scalar_decay",
            "lightnet_no_tpe_scalar_decay",
            "mla",
            "tnl",
            "tnl_state",
            "linear_transformer",
            "linear_transformer_no_tpe",
            "cosformer2",
            "cosformer2_no_tpe",
            "naive_deltanet",
            "scalar_decay_deltanet",
            "scalar_decay_lower_bound_deltanet",
            "vector_decay_deltanet",
            "vector_decay_lower_bound_deltanet",
            "dense_rnn",
            "dense_rnn_lower_bound",
            "decay_linear_transformer_hgrn2",
            "decay_linear_transformer_hgrn2_rope",
            "decay_linear_transformer_hgrn2_scalar_decay",
            "decay_linear_transformer_hgrn2_rope_scalar_decay",
            "decay_linear_transformer_mamba",
            "decay_linear_transformer_mamba_scalar_decay",
            "decay_linear_transformer_gla",
            "decay_linear_transformer_gla_scalar_decay",
            "decay_linear_transformer_lightnet",
            "decay_linear_transformer_lightnet_share_decay",
            "decay_linear_transformer_lightnet_scalar_decay",
            "decay_linear_transformer_lssp",
            "decay_linear_transformer_lssp_scalar_decay",
            "decay_linear_transformer_tnl",
            "decay_linear_transformer_tnl_scalar_decay",
            "nsa",
            "gsa",
            "ttt",
            "hgrn3",
            "hgrn3_scalar_decay",
            "alibi",
            "fox",
            "fox_window",
            "sb_attn",
            "mfa",
            "mfa_kv_share",
            "hgrn1",
            "mamba2",
            "scalar_decay_delta_product_net",
            "implicit_value_attn",
            "mesa_net",
            "poly_net",
            "kernel_regression_attn",
            "kernel_regression_attn_no_decay",
            "kernel_regression_attn_no_kr",
            "kernel_regression_attn_qk_norm",
            "path_attn",
            "path_attn_no_decay",
            "path_attn_no_beta",
            "fox_offset",
        ],
    )
    args = parser.parse_args()
    main(args)
