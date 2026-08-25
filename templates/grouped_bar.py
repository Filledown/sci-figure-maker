from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figure_style import apply_publication_style, clean_axes, get_palette
from templates.base import require_columns, save_figure


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
    """Grouped bar chart for model/dataset/metric comparisons."""
    require_columns(data, [category, *values])
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(3.65, 2.65))
    x = np.arange(len(data))
    n = len(values)
    width = min(0.74 / max(n, 1), 0.20)
    colors = get_palette(n)
    center = (n - 1) / 2

    for i, column in enumerate(values):
        ax.bar(
            x + (i - center) * width,
            data[column],
            width=width * 0.92,
            color=colors[i],
            edgecolor="none",
            label=column,
            zorder=3,
        )

    ax.set_xticks(x, data[category])
    if title:
        ax.set_title(title, loc="left", fontweight="semibold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    clean_axes(ax, grid_y=True)
    ax.legend(frameon=False, ncol=min(3, n), loc="best")
    fig.tight_layout()
    return save_figure(fig, output_dir)
