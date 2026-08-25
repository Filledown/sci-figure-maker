from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def save_figure(
    fig: plt.Figure,
    output_dir: str | Path,
    stem: str = "figure",
    dpi: int = 600,
) -> dict[str, Path]:
    """Save a figure as publication-ready PNG, SVG and PDF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "png": output_dir / f"{stem}.png",
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
    }

    fig.savefig(paths["png"], dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def require_columns(data, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
