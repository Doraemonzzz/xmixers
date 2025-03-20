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
    model_types = df["model_type"].unique()

    # Create figure
    plt.figure(figsize=(12, 8))

    # Color palette for different model types
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

    # Set labels and title
    plt.xlabel("Layer Index", fontsize=24)
    plt.ylabel("Median Value", fontsize=24)
    plt.title("Median Values Across Layers for Different Methods", fontsize=24)

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set x-ticks to show every other layer
    max_layer = int(df["layer_idx"].max())
    plt.xticks(range(1, max_layer + 1, 2))
    plt.tick_params(axis="both", which="major", labelsize=20)

    # Set y-axis limits
    plt.ylim(0, 1.3)

    # Add legend
    # plt.legend(fontsize=18, loc='best')
    # plt.legend(fontsize=14, loc='lower left')
    plt.legend(fontsize=14, loc="upper left")

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
        "dlt_mamba": "Mamba",
        "dlt_lightnet": "LightNet",
        "dlt_mamba_no_a": "Mamba W/O A",
        "dlt_mamba_no_a_no_t": "Mamba W/O A & T",
        "dlt_mamba_no_t": "Mamba W/O T",
        "dlt_scalar_gla": "GLA",
        "dlt_scalar_hgrn2": "HGRN2",
        "dlt_scalar_mamba": "Mamba",
        "dlt_tnl": "TNL",
        "dlt_tnll": "TNL-L",
        "dlt_share_gla": "GLA-Share",
        "dlt_share_hgrn2": "HGRN2-Share",
        "dlt_share_mamba": "Mamba-Share",
    }

    # Plot the data
    plot_model_median_by_layer(
        data_path=args.data_path,
        output_path=args.output_path,
        column=args.column,
        model_name_mapping=model_name_mapping,
    )
