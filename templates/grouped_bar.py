from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figure_style import (
    apply_publication_style,
    clean_axes,
    get_palette,
)
from templates.base import (
    require_columns,
    save_figure,
)


def render_grouped_bar(
    data: pd.DataFrame,
    category: str,
    values: Sequence[str],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    output_dir: str | Path = "outputs/grouped_bar",
) -> dict[str, Path]:
    """
    Grouped bar chart for model / dataset /
    metric comparisons.

    Layout rules:
    - legend stays on one row
    - legend is separated from the plotting area
    - title and legend do not overlap
    - enough top margin is reserved automatically
    """

    require_columns(
        data,
        [
            category,
            *values,
        ],
    )

    apply_publication_style()

    # Slightly wider than before.
    # Four metric names need enough horizontal room.
    fig, ax = plt.subplots(
        figsize=(4.05, 2.80)
    )

    x = np.arange(
        len(data)
    )

    n = len(values)

    if n == 0:
        raise ValueError(
            "Grouped bar requires at least "
            "one value column."
        )

    width = min(
        0.74 / n,
        0.20,
    )

    colors = get_palette(n)

    center = (
        n - 1
    ) / 2

    for i, column in enumerate(values):

        ax.bar(
            x
            + (i - center) * width,
            data[column],
            width=width * 0.92,
            color=colors[i],
            edgecolor="none",
            label=column,
            zorder=3,
        )

    ax.set_xticks(
        x,
        data[category],
    )

    if title:

        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
            pad=30,
        )

    if xlabel:

        ax.set_xlabel(
            xlabel
        )

    if ylabel:

        ax.set_ylabel(
            ylabel
        )

    clean_axes(
        ax,
        grid_y=True,
    )

    # Leave a little breathing room above bars.
    ax.margins(
        y=0.08
    )

    # --------------------------------------------------------
    # Legend layout
    # --------------------------------------------------------
    #
    # Important:
    # ncol=n prevents this:
    #
    # Precision  mAP50  mAP50_95
    #      Recall
    #
    # Instead all metrics remain in one clean row.
    #
    ax.legend(
        frameon=False,
        ncol=n,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            1.015,
        ),
        borderaxespad=0.0,
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=0.9,
    )

    fig.tight_layout(
        pad=0.8
    )

    return save_figure(
        fig,
        output_dir,
    )