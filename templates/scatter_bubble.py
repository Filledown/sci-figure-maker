from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figure_style import (
    ACCENT,
    PASTEL_5,
    apply_publication_style,
    clean_axes,
)
from templates.base import (
    require_columns,
    save_figure,
)


def _scale_bubbles(
    values: pd.Series,
    minimum: float = 38,
    maximum: float = 95,
) -> np.ndarray:
    """
    Compress bubble-size differences.

    Square-root scaling prevents a very large
    model from visually dominating the figure.
    """

    numeric = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(float)
    )

    transformed = np.sqrt(
        np.maximum(
            numeric,
            0,
        )
    )

    low = transformed.min()
    high = transformed.max()

    if high == low:

        return np.full(
            len(values),
            (
                minimum
                + maximum
            )
            / 2,
        )

    return (
        minimum
        + (
            transformed - low
        )
        / (
            high - low
        )
        * (
            maximum - minimum
        )
    )


def render_scatter_bubble(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    label: str | None = None,
    size: str | None = None,
    highlight: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    size_legend_title: str | None = None,
    output_dir: str | Path = (
        "outputs/scatter_bubble"
    ),
) -> dict[str, Path]:
    """
    Clean bubble scatter for accuracy-efficiency
    and other scientific trade-off plots.

    The bubble-size legend is deliberately placed
    outside the data region to avoid collisions
    with model labels.
    """

    required = [
        x,
        y,
    ]

    if label:
        required.append(label)

    if size:
        required.append(size)

    require_columns(
        data,
        required,
    )

    apply_publication_style()

    fig, ax = plt.subplots(
        figsize=(3.85, 2.70)
    )

    if size:

        sizes = _scale_bubbles(
            data[size]
        )

    else:

        sizes = np.full(
            len(data),
            48.0,
        )

    baseline_color = (
        PASTEL_5[2]
    )

    for position, (_, row) in enumerate(
        data.iterrows()
    ):

        item_label = (
            str(row[label])
            if label
            else None
        )

        selected = (
            highlight is not None
            and item_label == highlight
        )

        ax.scatter(
            row[x],
            row[y],
            s=sizes[position],
            color=(
                ACCENT
                if selected
                else baseline_color
            ),
            alpha=(
                0.95
                if selected
                else 0.78
            ),
            edgecolor="white",
            linewidth=0.65,
            zorder=(
                4
                if selected
                else 3
            ),
        )

        if item_label:

            # Highlighted model gets slightly more
            # separation from its point.
            offset = (
                (6, 5)
                if selected
                else (5, 3)
            )

            ax.annotate(
                item_label,
                (
                    row[x],
                    row[y],
                ),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.4,
                fontweight=(
                    "bold"
                    if selected
                    else "normal"
                ),
                zorder=5,
            )

    # --------------------------------------------------------
    # Bubble-size legend
    # --------------------------------------------------------

    if size:

        raw = pd.to_numeric(
            data[size],
            errors="coerce",
        ).dropna()

        if len(raw):

            legend_values = sorted(
                {
                    float(raw.min()),
                    float(raw.median()),
                    float(raw.max()),
                }
            )

            handles = []

            for value in legend_values:

                fake = pd.Series(
                    [
                        raw.min(),
                        value,
                        raw.max(),
                    ]
                )

                scaled = (
                    _scale_bubbles(
                        fake
                    )[1]
                )

                handles.append(
                    ax.scatter(
                        [],
                        [],
                        s=scaled,
                        color=baseline_color,
                        alpha=0.78,
                        edgecolor="white",
                        linewidth=0.6,
                    )
                )

            # Important change:
            #
            # legend is OUTSIDE the plotting area.
            #
            # Therefore Ours / baseline labels can
            # never collide with the size legend.
            ax.legend(
                handles,
                [
                    f"{value:g}"
                    for value
                    in legend_values
                ],
                title=(
                    size_legend_title
                    or size
                ),
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(
                    1.015,
                    1.0,
                ),
                borderaxespad=0.0,
                labelspacing=0.55,
                handletextpad=0.65,
            )

    if title:

        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
        )

    ax.set_xlabel(
        xlabel or x
    )

    ax.set_ylabel(
        ylabel or y
    )

    clean_axes(ax)

    ax.margins(
        x=0.10,
        y=0.12,
    )

    fig.tight_layout(
        pad=0.8
    )

    return save_figure(
        fig,
        output_dir,
    )