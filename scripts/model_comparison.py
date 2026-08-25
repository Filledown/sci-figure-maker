from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.load_data import load_data
from scripts.theme import (
    get_figure_width,
    get_semantic_palette,
    load_default_preset,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def validate_metric(
    data: pd.DataFrame,
    metric: str,
) -> None:
    """
    Check whether the requested metric exists
    and contains numeric values.
    """

    if metric not in data.columns:
        available = ", ".join(data.columns)

        raise KeyError(
            f"Metric '{metric}' was not found. "
            f"Available columns: {available}"
        )

    if not pd.api.types.is_numeric_dtype(data[metric]):
        raise TypeError(
            f"Metric '{metric}' must contain numeric data."
        )


def create_model_comparison(
    data: pd.DataFrame,
    metric: str,
    highlight: str | None = None,
    xlabel: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Create a publication-oriented horizontal model comparison plot.

    Baseline models use restrained neutral colors.
    The highlighted model receives a stronger accent color.
    """

    validate_metric(data, metric)

    preset = load_default_preset()

    semantic_palette = get_semantic_palette(
        "baseline_vs_ours"
    )

    baseline_colors = semantic_palette["baseline"]
    highlight_color = semantic_palette["highlight"]

    # Sort from lower to higher metric value.
    plot_data = (
        data[["Model", metric]]
        .copy()
        .sort_values(metric, ascending=True)
        .reset_index(drop=True)
    )

    models = plot_data["Model"].astype(str).tolist()
    values = plot_data[metric].astype(float).tolist()

    colors = []

    for index, model in enumerate(models):

        if highlight is not None and model == highlight:
            colors.append(highlight_color)
        else:
            colors.append(
                baseline_colors[
                    index % len(baseline_colors)
                ]
            )

    width = get_figure_width("single_column")

    # Height adapts slightly to the number of models.
    height = max(
        2.5,
        0.45 * len(models) + 1.2,
    )

    fig, ax = plt.subplots(
        figsize=(width, height)
    )

    y_positions = range(len(models))

    bars = ax.barh(
        y_positions,
        values,
        color=colors,
        height=0.62,
        edgecolor="none",
    )

    ax.set_yticks(
        list(y_positions),
        labels=models,
    )

    ax.set_xlabel(
        xlabel if xlabel else metric
    )

    # Clean publication-style axes.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(
        preset["axes"]["spine_width"]
    )

    ax.spines["bottom"].set_linewidth(
        preset["axes"]["spine_width"]
    )

    ax.tick_params(
        axis="both",
        labelsize=preset["typography"]["font_sizes"]["tick_label"],
        width=preset["axes"]["tick_width"],
        length=preset["axes"]["tick_length"],
    )

    ax.xaxis.label.set_size(
        preset["typography"]["font_sizes"]["axis_label"]
    )

    # Do not force the x-axis to start at an arbitrary high value.
    # Starting from zero avoids exaggerating small differences.
    ax.set_xlim(
        0,
        max(values) * 1.12
    )

    # Add exact values.
    for bar, value in zip(bars, values):

        ax.text(
            value + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=preset["typography"]["font_sizes"]["annotation"],
        )

    fig.tight_layout()

    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / "model_comparison"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = output_dir / "figure.png"
    svg_path = output_dir / "figure.svg"
    pdf_path = output_dir / "figure.pdf"

    dpi = preset["export"]["png"]["dpi"]

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return png_path


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-quality "
            "model comparison figure."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file.",
    )

    parser.add_argument(
        "--metric",
        required=True,
        help="Metric column to plot.",
    )

    parser.add_argument(
        "--highlight",
        default=None,
        help=(
            "Model name to emphasize, "
            "for example: Ours"
        ),
    )

    parser.add_argument(
        "--xlabel",
        default=None,
        help="Custom x-axis label.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/model_comparison",
        help="Directory for generated figures.",
    )

    return parser


def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    input_path = Path(args.input)

    output_dir = Path(args.output_dir)

    data = load_data(input_path)

    output_path = create_model_comparison(
        data=data,
        metric=args.metric,
        highlight=args.highlight,
        xlabel=args.xlabel,
        output_dir=output_dir,
    )

    print(
        "Model comparison figure generated successfully."
    )

    print(
        f"PNG: {output_path}"
    )

    print(
        f"SVG: {output_path.with_suffix('.svg')}"
    )

    print(
        f"PDF: {output_path.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()