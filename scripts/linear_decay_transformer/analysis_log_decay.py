import os

import numpy as np


def analyze_log_decay(base_dir, model_type, model_name):
    """
    Analyze log decay data for all layers and generate visualizations

    Parameters:
    base_dir: Base directory containing all layer data
    """
    print("model_type model_name layer_idx median 1_4_per 3_4_per mean max min std")
    total_data = []
    for file in os.listdir(base_dir):
        if file.endswith(".npy"):
            layer_idx = int(file.split("_")[-1].split(".")[0])
            file_path = os.path.join(base_dir, file)
            data = np.exp(np.load(file_path))[0]
            total_data.append(data)

            print(
                model_type,
                model_name,
                layer_idx,
                np.median(data),
                np.percentile(data, 25),
                np.percentile(data, 75),
                np.mean(data),
                np.max(data),
                np.min(data),
                np.std(data),
            )

    total_data = np.concatenate(total_data)
    print(
        model_type,
        model_name,
        -1,
        np.median(total_data),
        np.percentile(total_data, 25),
        np.percentile(total_data, 75),
        np.mean(total_data),
        np.max(total_data),
        np.min(total_data),
        np.std(total_data),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze log decay data")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing all layer data",
    )
    parser.add_argument("--model_type", type=str, required=True, help="Model type")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")

    args = parser.parse_args()

    stats_df = analyze_log_decay(args.base_dir, args.model_type, args.model_name)
