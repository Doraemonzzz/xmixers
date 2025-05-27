import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()


def plot_speed(
    data_path,
    output_path=None,
    model_name_mapping=None,
    selected_models=None,
    stage="fwd",
    data_type="speed",
):
    """
    Plot the speed comparison of different models across various sequence lengths.

    Args:
        data_path (str): Path to the CSV data file
        output_path (str, optional): Path to save the output figure
        model_name_mapping (dict, optional): Dictionary mapping model_type values to display names
        selected_models (list, optional): List of model names to plot, if None, plot all models
        stage (str, optional): "fwd" or "bwd"
        data_type (str, optional): "speed" or "memory"
    """
    # Read the data
    df = pd.read_csv(data_path, sep="\t")

    # Rename the first column to 'seq_len' for clarity
    col_names = df.columns.tolist()
    df.rename(columns={col_names[0]: "seq_len"}, inplace=True)

    # If selected_models is provided, filter columns
    if selected_models:
        # Ensure 'seq_len' is included and all selected models exist in the data
        available_cols = ["seq_len"]
        for model in selected_models:
            if model in df.columns:
                available_cols.append(model)
            else:
                print(f"Warning: Model '{model}' not found in data, skipping it.")

        # Filter DataFrame to include only selected models
        df = df[available_cols]

    # Convert data to long format for easier plotting with seaborn
    df_melted = df.melt(id_vars=["seq_len"], var_name="model_type", value_name="speed")

    # Define the desired order of model types
    model_order = [
        "MAMBA2",
        "MLSTM",
        "GLA_K",
        "GLA_S_K",
        "LAER_R",
        "LAER_P",
        "FLA_LAER",
        "LP",
        "LC",
        "LND_P",
        "LND_C",
        "Flash",
        "LACD_R",
        "LACD_P",
        "LAND_P",
        "LACD_PL",
        "LASD_P",
        "LAVD_K_P",
        "LAVD_KV_P",
    ]

    # Ensure all models in model_order exist in the data
    model_types = [m for m in model_order if m in df_melted["model_type"].unique()]

    # Add any models from the data that aren't in model_order
    missing_models = [
        m for m in df_melted["model_type"].unique() if m not in model_types
    ]
    model_types.extend(missing_models)

    # Create figure
    plt.figure(figsize=(14, 10))

    # Define colors
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
        "black",
        "teal",
        "navy",
        "lime",
        "coral",
        "gold",
        "orchid",
    ]

    # Plot each model's speed/memory curve
    for i, model_type in enumerate(model_types):
        model_data = df_melted[df_melted["model_type"] == model_type]
        color = colors[i % len(colors)]

        display_name = (
            model_name_mapping.get(model_type, model_type)
            if model_name_mapping
            else model_type
        )

        if data_type == "speed":
            data = model_data["speed"]
        else:
            data = model_data["speed"] / 1024

        plt.plot(
            model_data["seq_len"],
            data,
            "o-",
            label=display_name,
            color=color,
            linewidth=3,
            markersize=16,
        )

    # Set axis scales and labels
    plt.xscale("log", base=2)
    plt.xlabel("Sequence Length (log scale)", fontsize=40)
    if stage == "fwd":
        prefix = "Forward Pass"
    else:
        prefix = "Backward Pass"

    if data_type == "speed":
        plt.ylabel(f"{prefix} Time (ms)", fontsize=40)
        plt.title(f"{prefix} Speed Comparison", fontsize=40)
    else:
        plt.ylabel(f"{prefix} Memory (GB)", fontsize=40)
        plt.title(f"{prefix} Memory Comparison", fontsize=40)

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set tick label size
    plt.tick_params(axis="both", which="major", labelsize=30)

    # Add legend
    plt.legend(fontsize=40, loc="upper left")

    # Adjust layout to ensure legend isn't clipped
    plt.tight_layout()

    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Plot speed comparison of different models across sequence lengths"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the CSV data file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output figure (optional)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of model names to plot (optional, if not specified all models are plotted)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="fwd",
        help="Stage to plot (fwd or bwd)",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="speed",
        help="Data type to plot (speed or memory)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Model name mapping
    model_name_mapping = {
        "MAMBA2": "Mamba2",
        "GLA_K": "GLA",
        "GLA_S_K": "GLA-Sd",
        "LAER_R": "LA-Ele-R",
        "LAER_P": "LA-Ele",
        "FLA_LAER": "FLA-LAER",
        "LP": "Lightning",
        "LC": "Lightning-C",
        "LND_P": "Lightning-Nd",
        "LND_C": "Lightning-Nd-C",
        "MLSTM": "xLSTM",
        "Flash": "FA",
        "LACD_R": "ULA-Cd-R",
        "LACD_P": "ULA-Cd",
        "LAND_P": "ULA-Nd",
        "LACD_PL": "ULA-Cd-PL",
        "LASD_P": "ULA-Sd",
        "LAVD_K_P": "ULA-Vd",
        "LAVD_KV_P": "ULA-Vd-KV",
    }

    # Plot the data
    plot_speed(
        data_path=args.data_path,
        output_path=args.output_path,
        model_name_mapping=model_name_mapping,
        selected_models=args.models,
        stage=args.stage,
        data_type=args.data_type,
    )
