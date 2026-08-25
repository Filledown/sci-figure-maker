from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figure_style import ACCENT, LIGHT_GRAY, PASTEL_5, apply_publication_style, clean_axes
from templates.base import require_columns, save_figure


def render_lollipop(
    data: pd.DataFrame,
    category: str,
    value: str,
    *,
    highlight: str | None = None,
    sort: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    output_dir: str | Path = "outputs/lollipop",
) -> dict[str, Path]:
    """Lollipop ranking for ablation gain, per-class AP and model ranking."""
    require_columns(data, [category, value])
    apply_publication_style()

    plot_data = data[[category, value]].copy()
    plot_data[value] = pd.to_numeric(plot_data[value], errors="coerce")
    plot_data = plot_data.dropna()
    if sort:
        plot_data = plot_data.sort_values(value, ascending=False)

    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    y = np.arange(len(plot_data))

    for i, (_, row) in enumerate(plot_data.iterrows()):
        selected = highlight is not None and str(row[category]) == highlight
        ax.hlines(y[i], 0, row[value], color=LIGHT_GRAY, linewidth=1.4, zorder=1)
        ax.scatter(
            row[value], y[i],
            s=43,
            color=ACCENT if selected else PASTEL_5[2],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_yticks(y, plot_data[category].astype(str))
    ax.invert_yaxis()
    if title:
        ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel(xlabel or value)

    clean_axes(ax, grid_x=True)
    fig.tight_layout()
    return save_figure(fig, output_dir)
