from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from scripts.figure_style import (
    ACCENT,
    apply_publication_style,
    clean_axes,
    get_palette,
)
from templates.base import (
    require_columns,
    save_figure,
)


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
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    reference_diagonal: bool = False,
    sort_x: bool = False,
    output_dir: str | Path = "outputs/line",
) -> dict[str, Path]:
    """
    Render a publication-style multi-line scientific figure.

    This remains a generic visual template.

    Scientific semantics such as ROC AUC calculation are
    handled upstream by the dispatcher.
    """

    require_columns(
        data,
        [
            x,
            *ys,
        ],
    )

    apply_publication_style()

    labels = (
        list(labels)
        if labels is not None
        else list(ys)
    )

    if len(labels) != len(ys):

        raise ValueError(
            "labels and ys must have "
            "the same length."
        )

    # --------------------------------------------------------
    # Plot data
    # --------------------------------------------------------

    plot_data = data.copy()

    if sort_x:

        plot_data = (
            plot_data
            .sort_values(
                by=x,
                kind="stable",
            )
        )

    fig, ax = plt.subplots(
        figsize=(
            3.45,
            2.75,
        )
    )

    palette = get_palette(
        len(ys)
    )

    # --------------------------------------------------------
    # Optional scientific reference diagonal
    #
    # Used by ROC figures.
    # --------------------------------------------------------

    if reference_diagonal:

        ax.plot(
            [
                0.0,
                1.0,
            ],
            [
                0.0,
                1.0,
            ],
            linestyle="--",
            linewidth=0.8,
            color="#B9B9B9",
            alpha=0.85,
            zorder=1,
        )

    # --------------------------------------------------------
    # Data series
    # --------------------------------------------------------

    for i, (
        column,
        label,
    ) in enumerate(
        zip(
            ys,
            labels,
        )
    ):

        label_text = str(
            label
        )

        column_text = str(
            column
        )

        selected = False

        if highlight is not None:

            highlight_text = str(
                highlight
            )

            selected = (
                label_text
                == highlight_text
                or column_text
                == highlight_text
                or label_text.startswith(
                    f"{highlight_text} "
                )
            )

        ax.plot(
            plot_data[x],
            plot_data[column],
            marker="o",
            markersize=(
                3.0
                if selected
                else 2.6
            ),
            linewidth=(
                1.65
                if selected
                else 1.15
            ),
            color=(
                ACCENT
                if selected
                else palette[i]
            ),
            label=label_text,
            zorder=(
                5
                if selected
                else 3
            ),
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if title:

        ax.set_title(
            title,
            loc="left",
            fontweight="bold",
        )

    ax.set_xlabel(
        xlabel
        or x
    )

    if ylabel:

        ax.set_ylabel(
            ylabel
        )

    # --------------------------------------------------------
    # Scientific axis constraints
    # --------------------------------------------------------

    if xlim is not None:

        ax.set_xlim(
            xlim
        )

    if ylim is not None:

        ax.set_ylim(
            ylim
        )

    # --------------------------------------------------------
    # Publication styling
    # --------------------------------------------------------

    clean_axes(
        ax
    )

    if ys:

        ax.legend(
            frameon=False,
            loc="best",
            handlelength=1.8,
        )

    fig.tight_layout()

    return save_figure(
        fig,
        output_dir,
    )