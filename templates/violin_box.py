from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figure_style import apply_publication_style, clean_axes, get_palette
from templates.base import require_columns, save_figure


def render_violin_box(
    data: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | None = None,
    wide_columns: Sequence[str] | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    output_dir: str | Path = "outputs/violin_box",
) -> dict[str, Path]:
    """Violin + box plot; supports long or wide data."""
    apply_publication_style()

    if wide_columns:
        require_columns(data, wide_columns)
        labels = list(wide_columns)
        arrays = [pd.to_numeric(data[c], errors="coerce").dropna().to_numpy() for c in wide_columns]
    else:
        if not value or not group:
            raise ValueError("Provide value+group or wide_columns.")
        require_columns(data, [value, group])
        labels = [str(x) for x in pd.unique(data[group].dropna())]
        arrays = [
            pd.to_numeric(data.loc[data[group] == label, value], errors="coerce").dropna().to_numpy()
            for label in labels
        ]

    if not arrays or any(len(arr) == 0 for arr in arrays):
        raise ValueError("Each violin group must contain at least one numeric value.")

    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    colors = get_palette(len(arrays))

    violins = ax.violinplot(arrays, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.88)

    ax.boxplot(
        arrays,
        widths=0.12,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#555555", "linewidth": 0.9},
        whiskerprops={"color": "#666666", "linewidth": 0.7},
        capprops={"color": "#666666", "linewidth": 0.7},
        boxprops={"facecolor": "white", "edgecolor": "#555555", "linewidth": 0.7},
    )

    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    if title:
        ax.set_title(title, loc="left", fontweight="semibold")
    if ylabel:
        ax.set_ylabel(ylabel)

    clean_axes(ax)
    fig.tight_layout()
    return save_figure(fig, output_dir)
