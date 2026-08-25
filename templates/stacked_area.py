from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from scripts.figure_style import apply_publication_style, clean_axes, get_palette
from templates.base import require_columns, save_figure


def render_stacked_area(
    data: pd.DataFrame,
    x: str,
    ys: Sequence[str],
    *,
    normalize: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    output_dir: str | Path = "outputs/stacked_area",
) -> dict[str, Path]:
    """Stacked area chart for category/error/scenario composition over a continuous axis."""
    require_columns(data, [x, *ys])
    apply_publication_style()

    values = data[list(ys)].apply(pd.to_numeric, errors="coerce").fillna(0).copy()
    if normalize:
        totals = values.sum(axis=1).replace(0, 1)
        values = values.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(3.65, 2.6))
    ax.stackplot(
        data[x],
        *[values[c] for c in ys],
        labels=list(ys),
        colors=get_palette(len(ys)),
        alpha=0.92,
        linewidth=0.45,
    )

    if title:
        ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or ("Proportion (%)" if normalize else "Value"))

    clean_axes(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return save_figure(fig, output_dir)
