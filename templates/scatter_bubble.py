from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figure_style import ACCENT, PASTEL_5, apply_publication_style, clean_axes
from templates.base import require_columns, save_figure


def _scale_bubbles(values: pd.Series, minimum: float = 38, maximum: float = 95) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(float)
    transformed = np.sqrt(np.maximum(numeric, 0))
    lo, hi = transformed.min(), transformed.max()
    if hi == lo:
        return np.full(len(values), (minimum + maximum) / 2)
    return minimum + (transformed - lo) / (hi - lo) * (maximum - minimum)


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
    output_dir: str | Path = "outputs/scatter_bubble",
) -> dict[str, Path]:
    """Clean bubble scatter for accuracy-efficiency and trade-off plots."""
    required = [x, y]
    if label:
        required.append(label)
    if size:
        required.append(size)
    require_columns(data, required)
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    sizes = _scale_bubbles(data[size]) if size else np.full(len(data), 48.0)
    baseline_color = PASTEL_5[2]

    for pos, (_, row) in enumerate(data.iterrows()):
        item_label = str(row[label]) if label else None
        selected = highlight is not None and item_label == highlight

        ax.scatter(
            row[x],
            row[y],
            s=sizes[pos],
            color=ACCENT if selected else baseline_color,
            alpha=0.95 if selected else 0.78,
            edgecolor="white",
            linewidth=0.65,
            zorder=4 if selected else 3,
        )

        if item_label:
            ax.annotate(
                item_label,
                (row[x], row[y]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.4,
                fontweight="semibold" if selected else "normal",
            )

    if size:
        raw = pd.to_numeric(data[size], errors="coerce").dropna()
        if len(raw):
            legend_values = sorted({float(raw.min()), float(raw.median()), float(raw.max())})
            handles = []
            for value in legend_values:
                fake = pd.Series([raw.min(), value, raw.max()])
                scaled = _scale_bubbles(fake)[1]
                handles.append(ax.scatter([], [], s=scaled, color=baseline_color, alpha=0.78, edgecolor="white"))
            ax.legend(
                handles,
                [f"{v:g}" for v in legend_values],
                title=size_legend_title or size,
                frameon=False,
                loc="best",
                borderpad=0.2,
                labelspacing=0.5,
            )

    if title:
        ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)

    clean_axes(ax)
    ax.margins(x=0.10, y=0.12)
    fig.tight_layout()
    return save_figure(fig, output_dir)
