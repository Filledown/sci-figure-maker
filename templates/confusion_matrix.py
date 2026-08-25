from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from scripts.figure_style import apply_publication_style
from templates.base import save_figure


CMAP = LinearSegmentedColormap.from_list(
    "sci_confusion",
    ["#F7FBFF", "#B3CDE4", "#5A8FC4"],
)


def _normalize(matrix: np.ndarray, mode: str) -> np.ndarray:
    matrix = matrix.astype(float)
    if mode == "none":
        return matrix
    if mode == "row":
        denom = matrix.sum(axis=1, keepdims=True)
    elif mode == "column":
        denom = matrix.sum(axis=0, keepdims=True)
    elif mode == "all":
        denom = np.array([[matrix.sum()]])
    else:
        raise ValueError("normalize must be one of: none, row, column, all")
    return np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)


def render_confusion_matrix(
    matrix: pd.DataFrame | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    normalize: str = "row",
    title: str | None = None,
    output_dir: str | Path = "outputs/confusion_matrix",
) -> dict[str, Path]:
    """Confusion matrix with explicit normalization mode."""
    apply_publication_style()

    if isinstance(matrix, pd.DataFrame):
        if labels is None:
            labels = [str(x) for x in matrix.index]
        raw = matrix.to_numpy(dtype=float)
    else:
        raw = np.asarray(matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("Confusion matrix must be a square 2-D matrix.")

    if labels is None:
        labels = [str(i) for i in range(raw.shape[0])]
    if len(labels) != raw.shape[0]:
        raise ValueError("labels length must match matrix size.")

    values = _normalize(raw, normalize)
    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    image = ax.imshow(values, cmap=CMAP, aspect="equal")

    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    max_value = np.nanmax(values) if values.size else 1
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = f"{value:.2f}" if normalize != "none" else f"{int(round(value))}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=5.7,
                color="white" if max_value and value > max_value * 0.55 else "#222222",
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
