import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import xmixers  # noqa


def save_log_f(
    model, tokenizer, text, save_dir, save_name, max_length=2048, device="cuda"
):
    """
    Run model inference and save log_f values.

    Args:
        model: The model to extract log_f from
        tokenizer: Tokenizer for input processing
        text: Input text for inference
        save_dir: Directory to save log_f values
        save_name: Filename for saved data
        max_length: Maximum sequence length for tokenization
        device: Device to run inference on
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    os.path.join(save_dir, f"{save_name}.npy")

    # Tokenize input text with truncation
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    inputs["save_decay"] = True
    inputs["save_dir"] = save_dir
    inputs["save_name"] = save_name
    print(f"Input sequence length: {inputs['input_ids'].shape[1]}")

    # Run inference
    with torch.inference_mode():
        model(
            **inputs,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Save log_f values from decay linear attention models"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--text_file", type=str, required=True, help="Path to text file for input"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save log_f values"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
        help="Filename for saved data (without extension)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Data type for model",
    )
    args = parser.parse_args()

    # Setup device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Read input text from file
    print(f"Reading text from {args.text_file}")
    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract and save log_f values
    print(f"Processing text (max length: {args.max_length})")
    save_log_f(
        model, tokenizer, text, args.save_dir, args.save_name, args.max_length, device
    )


if __name__ == "__main__":
    main()
