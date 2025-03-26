import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()


def plot_model_median_by_layer(
    data_path, output_path=None, column="median", model_name_mapping=None
):
    """
    Plot the median values across layers for different model types.

    Args:
        data_path (str): Path to the tab-separated data file
        output_path (str, optional): Path to save the output figure
        model_name_mapping (dict, optional): Dictionary mapping model_type values to display names
    """
    # Read the data using pandas
    # Skip empty rows and use tab as separator
    df = pd.read_csv(data_path, sep="\t")

    # Filter out rows where layer_idx is -1
    df = df[df["layer_idx"] != -1]

    # Add 1 to layer_idx to start from 1 instead of 0
    df["layer_idx"] = df["layer_idx"] + 1

    # Sort by layer_idx to ensure correct ordering
    df = df.sort_values(["model_type", "layer_idx"])

    # Get unique model types for plotting with different colors
    all_model_types = df["model_type"].unique()

    # Define the desired order of model types (add all your model types in desired order)
    model_order = [
        "dlt_mamba",
        "dlt_gla",
        "dlt_hgrn2",
        "dlt_lightnet",
        "dlt_mamba_no_a",
        "dlt_mamba_no_t",
        "dlt_mamba_no_a_no_t",
        "dlt_scalar_mamba",
        "dlt_scalar_gla",
        "dlt_scalar_hgrn2",
        "dlt_scalar_lightnet",
        "dlt_tnl",
        "dlt_tnll",
        "dlt_share_mamba",
        "dlt_share_gla",
        "dlt_share_hgrn2",
        "dlt_share_lightnet",
        "dlt_hgrn3_0_8",
        "dlt_hgrn3_0_9",
        "dlt_hgrn3_0_95",
        "dlt_hgrn3_0_99",
    ]

    # Filter model_order to include only models present in the data
    model_types = [m for m in model_order if m in all_model_types]

    # Create figure
    plt.figure(figsize=(12, 8))

    # Color palette for different model types
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "gray",
        "purple",
        "brown",
        "pink",
        "olive",
        "cyan",
    ]

    # Plot each model type with a different color
    for i, model_type in enumerate(model_types):
        model_data = df[df["model_type"] == model_type]
        color = colors[
            i % len(colors)
        ]  # Cycle through colors if more models than colors
        display_name = (
            model_name_mapping.get(model_type, model_type)
            if model_name_mapping
            else model_type
        )
        plt.plot(
            model_data["layer_idx"],
            model_data[column],
            "o-",
            label=display_name,
            color=color,
            linewidth=4,
            markersize=10,
        )

    # Determine title suffix based on data_path filename
    title_suffix = ""
    if "90m" in data_path:
        title_suffix = " (160M)"
    elif "350m" in data_path:
        title_suffix = " (410M)"
    elif "1_2b" in data_path:
        title_suffix = " (1.45B)"

    # Set labels and title
    plt.xlabel("Layer Index", fontsize=28)
    plt.ylabel("Median Decay Value", fontsize=28)
    plt.title(
        f"Median Decay Values Across Layers\n for Different Methods{title_suffix}",
        fontsize=28,
    )

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set x-ticks to show every other layer
    max_layer = int(df["layer_idx"].max())
    plt.xticks(range(1, max_layer + 1, 2))
    plt.tick_params(axis="both", which="major", labelsize=24)

    # Set y-axis limits
    if "scalar" in data_path:
        plt.ylim(0, 1.55)
    else:
        plt.ylim(0, 1.4)

    # Add legend
    plt.legend(fontsize=16, loc="upper left")

    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    # Show plot
    plt.show()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Plot model median values across layers"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the tab-separated data file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output figure (optional)",
    )
    parser.add_argument(
        "--column", type=str, default="median", help="Column name to plot (optional)"
    )
    parser.add_argument(
        "--model_name_mapping",
        type=str,
        default=None,
        help="Path to the model name mapping file (optional)",
    )

    # Parse arguments
    args = parser.parse_args()

    model_name_mapping = {
        "dlt_gla": "GLA",
        "dlt_hgrn2": "HGRN2",
        "dlt_mamba": "Mamba2",
        "dlt_lightnet": "LightNet",
        "dlt_mamba_no_a": "Mamba2 W/O A",
        "dlt_mamba_no_a_no_t": "Mamba2 W/O A & Δ",
        "dlt_mamba_no_t": "Mamba2 W/O Δ",
        "dlt_scalar_gla": "GLA-Scalar",
        "dlt_scalar_hgrn2": "HGRN2-Scalar",
        "dlt_scalar_mamba": "Mamba2-Scalar",
        "dlt_scalar_lightnet": "LightNet-Scalar",
        "dlt_tnl": "TNL",
        "dlt_tnll": "TNL-L",
        "dlt_share_gla": "GLA-Share",
        "dlt_share_hgrn2": "HGRN2-Share",
        "dlt_share_mamba": "Mamba2-Share",
        "dlt_share_lightnet": "LightNet-Share",
        "dlt_hgrn3_0_8": "SD-0.8",
        "dlt_hgrn3_0_9": "SD-0.9",
        "dlt_hgrn3_0_95": "SD-0.95",
        "dlt_hgrn3_0_99": "SD-0.99",
    }

    # Plot the data
    plot_model_median_by_layer(
        data_path=args.data_path,
        output_path=args.output_path,
        column=args.column,
        model_name_mapping=model_name_mapping,
    )
