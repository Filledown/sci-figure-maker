from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from scripts.figure_style import apply_publication_style
from templates.base import require_columns, save_figure


SOFT_DIVERGING = LinearSegmentedColormap.from_list(
    "sci_soft_diverging",
    ["#B3CDE4", "#F7F7F7", "#FBB4AE"],
)


def render_heatmap(
    data: pd.DataFrame,
    *,
    row_label: str | None = None,
    value_columns: Sequence[str] | None = None,
    annotate: bool = True,
    fmt: str = ".2f",
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    output_dir: str | Path = "outputs/heatmap",
) -> dict[str, Path]:
    """Publication heatmap for robustness matrices and model-by-scenario tables."""
    apply_publication_style()

    if value_columns is None:
        value_columns = [c for c in data.columns if c != row_label and pd.api.types.is_numeric_dtype(data[c])]
    required = list(value_columns)
    if row_label:
        required.append(row_label)
    require_columns(data, required)

    matrix = data[list(value_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isnan(matrix).all():
        raise ValueError("Heatmap has no numeric values.")

    fig, ax = plt.subplots(figsize=(3.65, 2.75))
    image = ax.imshow(matrix, cmap=SOFT_DIVERGING, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(value_columns)), list(value_columns))
    if row_label:
        ax.set_yticks(np.arange(len(data)), data[row_label].astype(str))
    else:
        ax.set_yticks(np.arange(len(data)), [str(i + 1) for i in range(len(data))])

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    if annotate:
        finite = matrix[np.isfinite(matrix)]
        midpoint = np.nanmedian(finite) if finite.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(
                        j, i, format(value, fmt),
                        ha="center", va="center",
                        fontsize=6.0,
                        color="#333333",
                    )

    if title:
        ax.set_title(title, loc="left", fontweight="semibold")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    return save_figure(fig, output_dir)
