from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.load_data import load_data
from scripts.theme import (
    get_figure_width,
    get_semantic_palette,
    load_default_preset,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def validate_columns(
    data: pd.DataFrame,
    model_col: str,
    x_col: str,
    y_col: str,
    size_col: str | None,
) -> None:

    required = [model_col, x_col, y_col]

    if size_col is not None:
        required.append(size_col)

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "Missing required column(s): "
            + ", ".join(missing)
        )

    numeric_columns = [x_col, y_col]

    if size_col is not None:
        numeric_columns.append(size_col)

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            raise TypeError(
                f"Column '{column}' must be numeric."
            )


def scale_bubble_sizes(
    values: pd.Series,
    minimum: float = 48.0,
    maximum: float = 135.0,
) -> np.ndarray:
    """
    Compress bubble-size differences so very large
    models do not visually dominate the figure.
    """

    numeric = values.astype(float).to_numpy()

    value_min = numeric.min()
    value_max = numeric.max()

    if value_min == value_max:
        return np.full(
            len(numeric),
            (minimum + maximum) / 2,
        )

    # Square-root transformation reduces extreme
    # visual differences between model sizes.
    transformed = np.sqrt(numeric)

    transformed_min = transformed.min()
    transformed_max = transformed.max()

    scaled = (
        minimum
        + (
            (transformed - transformed_min)
            / (
                transformed_max
                - transformed_min
            )
        )
        * (maximum - minimum)
    )

    return scaled


def get_pareto_mask(
    x: np.ndarray,
    y: np.ndarray,
    x_direction: str = "higher",
    y_direction: str = "higher",
) -> np.ndarray:
    """
    Return True for non-dominated observations.

    Example:
        FPS: higher is better
        mAP: higher is better
    """

    x_score = (
        x
        if x_direction == "higher"
        else -x
    )

    y_score = (
        y
        if y_direction == "higher"
        else -y
    )

    n = len(x_score)

    pareto = np.ones(
        n,
        dtype=bool,
    )

    for i in range(n):

        for j in range(n):

            if i == j:
                continue

            at_least_as_good = (
                x_score[j] >= x_score[i]
                and y_score[j] >= y_score[i]
            )

            strictly_better = (
                x_score[j] > x_score[i]
                or y_score[j] > y_score[i]
            )

            if (
                at_least_as_good
                and strictly_better
            ):
                pareto[i] = False
                break

    return pareto


def choose_label_offset(
    x: float,
    y: float,
    x_mid: float,
    y_mid: float,
) -> tuple[int, int, str]:

    if x >= x_mid and y >= y_mid:
        return 6, 5, "left"

    if x >= x_mid and y < y_mid:
        return 6, -2, "left"

    if x < x_mid and y >= y_mid:
        return 6, 5, "left"

    return 6, -2, "left"


def create_accuracy_efficiency(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    model_col: str = "Model",
    size_col: str | None = None,
    highlight: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    x_direction: str = "higher",
    y_direction: str = "higher",
    output_dir: Path | None = None,
) -> Path:

    validate_columns(
        data=data,
        model_col=model_col,
        x_col=x_col,
        y_col=y_col,
        size_col=size_col,
    )

    preset = load_default_preset()

    semantic_palette = get_semantic_palette(
        "baseline_vs_ours"
    )

    baseline_color = semantic_palette[
        "baseline"
    ][0]

    highlight_color = semantic_palette[
        "highlight"
    ]

    columns = [
        model_col,
        x_col,
        y_col,
    ]

    if size_col is not None:
        columns.append(size_col)

    plot_data = (
        data[columns]
        .copy()
        .dropna()
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "No valid rows remain after "
            "removing missing values."
        )

    models = (
        plot_data[model_col]
        .astype(str)
        .to_numpy()
    )

    x_values = (
        plot_data[x_col]
        .astype(float)
        .to_numpy()
    )

    y_values = (
        plot_data[y_col]
        .astype(float)
        .to_numpy()
    )

    if size_col is not None:
        raw_sizes = (
            plot_data[size_col]
            .astype(float)
        )

        bubble_sizes = scale_bubble_sizes(
            raw_sizes
        )

    else:
        raw_sizes = None

        bubble_sizes = np.full(
            len(plot_data),
            72.0,
        )

    pareto_mask = get_pareto_mask(
        x=x_values,
        y=y_values,
        x_direction=x_direction,
        y_direction=y_direction,
    )

    width = get_figure_width(
        "single_column"
    )

    height = width * 0.78

    fig, ax = plt.subplots(
        figsize=(width, height)
    )

    # ---------------------------------------------------------
    # Baseline observations
    # ---------------------------------------------------------

    baseline_mask = np.array(
        [
            model != highlight
            for model in models
        ]
    )

    ax.scatter(
        x_values[baseline_mask],
        y_values[baseline_mask],
        s=bubble_sizes[baseline_mask],
        color=baseline_color,
        alpha=0.66,
        edgecolors="white",
        linewidths=0.65,
        zorder=3,
    )

    # ---------------------------------------------------------
    # Highlighted model
    # ---------------------------------------------------------

    if highlight is not None:

        highlight_mask = (
            models == highlight
        )

        if highlight_mask.any():

            ax.scatter(
                x_values[highlight_mask],
                y_values[highlight_mask],
                s=(
                    bubble_sizes[
                        highlight_mask
                    ]
                    * 1.08
                ),
                color=highlight_color,
                alpha=0.95,
                edgecolors="white",
                linewidths=0.8,
                zorder=6,
            )

    # ---------------------------------------------------------
    # Pareto frontier
    # ---------------------------------------------------------

    pareto_data = pd.DataFrame(
        {
            "x": x_values[
                pareto_mask
            ],
            "y": y_values[
                pareto_mask
            ],
        }
    )

    if len(pareto_data) >= 2:

        ascending = (
            x_direction == "higher"
        )

        pareto_data = (
            pareto_data
            .sort_values(
                "x",
                ascending=ascending,
            )
        )

        ax.plot(
            pareto_data["x"],
            pareto_data["y"],
            color=baseline_color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.42,
            zorder=2,
        )

    # ---------------------------------------------------------
    # Model labels
    # ---------------------------------------------------------

    x_mid = (
        x_values.min()
        + x_values.max()
    ) / 2

    y_mid = (
        y_values.min()
        + y_values.max()
    ) / 2

    annotation_size = max(
        6.0,
        preset[
            "typography"
        ][
            "font_sizes"
        ][
            "annotation"
        ]
        - 0.5,
    )

    for i, model in enumerate(models):

        dx, dy, alignment = (
            choose_label_offset(
                x_values[i],
                y_values[i],
                x_mid,
                y_mid,
            )
        )

        is_highlight = (
            highlight is not None
            and model == highlight
        )

        ax.annotate(
            model,
            (
                x_values[i],
                y_values[i],
            ),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=annotation_size,
            fontweight=(
                "semibold"
                if is_highlight
                else "normal"
            ),
            zorder=8,
        )

    # ---------------------------------------------------------
    # Axis styling
    # ---------------------------------------------------------

    ax.set_xlabel(
        xlabel
        if xlabel
        else x_col
    )

    ax.set_ylabel(
        ylabel
        if ylabel
        else y_col
    )

    axis_font = max(
        7.0,
        preset[
            "typography"
        ][
            "font_sizes"
        ][
            "axis_label"
        ]
        - 0.5,
    )

    tick_font = max(
        6.5,
        preset[
            "typography"
        ][
            "font_sizes"
        ][
            "tick_label"
        ]
        - 0.5,
    )

    ax.xaxis.label.set_size(
        axis_font
    )

    ax.yaxis.label.set_size(
        axis_font
    )

    ax.tick_params(
        axis="both",
        labelsize=tick_font,
        width=0.7,
        length=3.0,
        direction="out",
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.spines["left"].set_linewidth(
        0.75
    )

    ax.spines["bottom"].set_linewidth(
        0.75
    )

    # ---------------------------------------------------------
    # Bubble-size legend
    # ---------------------------------------------------------

    if (
        size_col is not None
        and raw_sizes is not None
    ):

        legend_values = np.unique(
            np.round(
                np.quantile(
                    raw_sizes,
                    [0.0, 0.5, 1.0],
                ),
                1,
            )
        )

        legend_handles = []

        for value in legend_values:

            scaled_size = (
                scale_bubble_sizes(
                    pd.Series(
                        [
                            raw_sizes.min(),
                            value,
                            raw_sizes.max(),
                        ]
                    )
                )[1]
            )

            handle = ax.scatter(
                [],
                [],
                s=scaled_size,
                color=baseline_color,
                alpha=0.55,
                edgecolors="white",
                linewidths=0.5,
                label=f"{value:g}",
            )

            legend_handles.append(
                handle
            )

        legend = ax.legend(
            handles=legend_handles,
            title=size_col,
            loc="lower left",
            frameon=False,
            fontsize=max(
                5.5,
                tick_font - 0.7,
            ),
            title_fontsize=max(
                6.0,
                tick_font - 0.3,
            ),
            borderaxespad=0.0,
            handletextpad=0.5,
            labelspacing=0.5,
        )

        ax.add_artist(
            legend
        )

    # ---------------------------------------------------------
    # Direction cue
    # ---------------------------------------------------------

    if (
        x_direction == "higher"
        and y_direction == "higher"
    ):

        ax.text(
            0.985,
            0.98,
            "Better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=max(
                5.5,
                annotation_size - 0.5,
            ),
            alpha=0.45,
        )

    # ---------------------------------------------------------
    # Margins
    # ---------------------------------------------------------

    x_range = (
        x_values.max()
        - x_values.min()
    )

    y_range = (
        y_values.max()
        - y_values.min()
    )

    if x_range == 0:
        x_range = 1

    if y_range == 0:
        y_range = 1

    ax.set_xlim(
        x_values.min()
        - x_range * 0.12,
        x_values.max()
        + x_range * 0.17,
    )

    ax.set_ylim(
        y_values.min()
        - y_range * 0.13,
        y_values.max()
        + y_range * 0.13,
    )

    fig.tight_layout()

    if output_dir is None:

        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / "accuracy_efficiency"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_dir
        / "figure.png"
    )

    svg_path = (
        output_dir
        / "figure.svg"
    )

    pdf_path = (
        output_dir
        / "figure.pdf"
    )

    dpi = preset[
        "export"
    ][
        "png"
    ][
        "dpi"
    ]

    for path in [
        png_path,
        svg_path,
        pdf_path,
    ]:

        save_kwargs = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }

        if (
            path.suffix.lower()
            == ".png"
        ):
            save_kwargs[
                "dpi"
            ] = dpi

        fig.savefig(
            path,
            **save_kwargs,
        )

    plt.close(fig)

    return png_path


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-quality "
            "accuracy-efficiency trade-off figure."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--sheet",
        default=0,
    )

    parser.add_argument(
        "--x",
        required=True,
    )

    parser.add_argument(
        "--y",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="Model",
    )

    parser.add_argument(
        "--size",
        default=None,
    )

    parser.add_argument(
        "--highlight",
        default=None,
    )

    parser.add_argument(
        "--xlabel",
        default=None,
    )

    parser.add_argument(
        "--ylabel",
        default=None,
    )

    parser.add_argument(
        "--x-direction",
        choices=[
            "higher",
            "lower",
        ],
        default="higher",
    )

    parser.add_argument(
        "--y-direction",
        choices=[
            "higher",
            "lower",
        ],
        default="higher",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/"
            "accuracy_efficiency"
        ),
    )

    return parser


def parse_sheet_argument(
    value: str,
) -> str | int:

    try:
        return int(value)

    except ValueError:
        return value


def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    data = load_data(
        Path(args.input),
        sheet_name=(
            parse_sheet_argument(
                str(args.sheet)
            )
        ),
    )

    output_path = (
        create_accuracy_efficiency(
            data=data,
            x_col=args.x,
            y_col=args.y,
            model_col=args.model,
            size_col=args.size,
            highlight=args.highlight,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            x_direction=args.x_direction,
            y_direction=args.y_direction,
            output_dir=Path(
                args.output_dir
            ),
        )
    )

    print(
        "Accuracy-efficiency figure "
        "generated successfully."
    )

    print(
        f"PNG: {output_path}"
    )

    print(
        "SVG: "
        f"{output_path.with_suffix('.svg')}"
    )

    print(
        "PDF: "
        f"{output_path.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()