from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figure_style import apply_publication_style, get_palette
from templates.base import require_columns, save_figure


def _smooth_hist(values: np.ndarray, bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("Each ridge group requires at least two numeric values.")

    hist, edges = np.histogram(values, bins=bins, density=True)
    x = (edges[:-1] + edges[1:]) / 2

    radius = 7
    grid = np.arange(-radius, radius + 1)
    sigma = 2.0
    kernel = np.exp(-(grid ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    smooth = np.convolve(hist, kernel, mode="same")
    if smooth.max() > 0:
        smooth = smooth / smooth.max()
    return x, smooth


def render_ridge(
    data: pd.DataFrame,
    group: str,
    value: str,
    *,
    reference: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    output_dir: str | Path = "outputs/ridge",
) -> dict[str, Path]:
    """Ridgeline-style density plot without SciPy dependency."""
    require_columns(data, [group, value])
    apply_publication_style()

    groups = [str(x) for x in pd.unique(data[group].dropna())]
    colors = get_palette(len(groups))
    fig, ax = plt.subplots(figsize=(3.6, 2.7))

    for i, label in enumerate(groups):
        values = pd.to_numeric(data.loc[data[group].astype(str) == label, value], errors="coerce").dropna().to_numpy(float)
        x, density = _smooth_hist(values)
        baseline = len(groups) - 1 - i
        height = density * 0.72

        ax.fill_between(
            x, baseline, baseline + height,
            color=colors[i],
            alpha=0.78,
            linewidth=0,
        )
        ax.plot(x, baseline + height, color=colors[i], linewidth=0.9)

    if reference is not None:
        ax.axvline(reference, color="#777777", linestyle="--", linewidth=0.7)

    ax.set_yticks(np.arange(len(groups)), list(reversed(groups)))
    ax.set_xlabel(xlabel or value)
    if title:
        ax.set_title(title, loc="left", fontweight="semibold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return save_figure(fig, output_dir)
