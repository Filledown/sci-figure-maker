from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Core palette
# ------------------------------------------------------------

PASTEL_5 = [
    "#FBB4AE",  # coral
    "#FFD9A8",  # warm yellow
    "#B3CDE4",  # soft blue
    "#CCEAC4",  # soft green
    "#DECAE5",  # lavender
]

PASTEL_10 = [
    "#FBB4AE",
    "#FFD9A8",
    "#B3CDE4",
    "#CCEAC4",
    "#DECAE5",
    "#F8A5A5",
    "#FDD084",
    "#8DC5D8",
    "#A8D5B0",
    "#CDB4E6",
]

ACCENT = "#E96B56"

TEXT = "#202020"
MUTED_TEXT = "#666666"
AXIS = "#444444"
GRID = "#E9E9E9"
LIGHT_GRAY = "#D8DEE4"


def apply_publication_style() -> None:
    """
    Apply the default sci-figure-maker visual style.
    """

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
            ],

            "font.size": 7.5,

            "axes.labelsize": 8,
            "axes.titlesize": 9,

            "xtick.labelsize": 7,
            "ytick.labelsize": 7,

            "legend.fontsize": 6.8,

            "axes.linewidth": 0.65,

            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,

            "xtick.major.size": 3,
            "ytick.major.size": 3,

            "axes.edgecolor": AXIS,
            "axes.labelcolor": TEXT,

            "xtick.color": TEXT,
            "ytick.color": TEXT,

            "text.color": TEXT,

            "figure.facecolor": "white",
            "axes.facecolor": "white",

            "savefig.facecolor": "white",
            "savefig.transparent": False,

            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            "svg.fonttype": "none",
        }
    )


def clean_axes(
    ax: plt.Axes,
    *,
    grid_y: bool = False,
    grid_x: bool = False,
) -> None:
    """
    Apply restrained publication-style axes.
    """

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)

    ax.tick_params(
        direction="out",
        width=0.6,
        length=3,
    )

    if grid_y:
        ax.grid(
            axis="y",
            color=GRID,
            linewidth=0.5,
            alpha=0.8,
            zorder=0,
        )

    if grid_x:
        ax.grid(
            axis="x",
            color=GRID,
            linewidth=0.5,
            alpha=0.8,
            zorder=0,
        )


def get_palette(
    n: int,
) -> list[str]:

    if n <= 5:
        palette = PASTEL_5
    else:
        palette = PASTEL_10

    return [
        palette[i % len(palette)]
        for i in range(n)
    ]