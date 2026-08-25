from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from scripts.figure_style import ACCENT, apply_publication_style, clean_axes, get_palette
from templates.base import require_columns, save_figure


def render_line(
    data: pd.DataFrame,
    x: str,
    ys: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    highlight: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    output_dir: str | Path = "outputs/line",
) -> dict[str, Path]:
    """Multi-line publication plot for training/metric curves."""
    require_columns(data, [x, *ys])
    apply_publication_style()

    labels = list(labels) if labels is not None else list(ys)
    if len(labels) != len(ys):
        raise ValueError("labels and ys must have the same length.")

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    palette = get_palette(len(ys))

    for i, (column, label) in enumerate(zip(ys, labels)):
        selected = highlight is not None and label == highlight
        ax.plot(
            data[x],
            data[column],
            marker="o",
            markersize=2.6,
            linewidth=1.5 if selected else 1.15,
            color=ACCENT if selected else palette[i],
            label=label,
            zorder=4 if selected else 3,
        )

    if title:
        ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel(xlabel or x)
    if ylabel:
        ax.set_ylabel(ylabel)

    clean_axes(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return save_figure(fig, output_dir)
