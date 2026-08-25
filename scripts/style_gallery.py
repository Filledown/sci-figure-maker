from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figure_style import (
    ACCENT,
    LIGHT_GRAY,
    PASTEL_5,
    apply_publication_style,
    clean_axes,
)


OUTPUT_DIR = Path(
    "outputs/style_gallery"
)


def main() -> None:

    apply_publication_style()

    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 6.3),
    )

    # ========================================================
    # 1. Multi-line
    # ========================================================

    ax = axes[0, 0]

    x = np.arange(0, 11)

    for i, color in enumerate(PASTEL_5):

        y = (
            10
            + i * 8
            + x * (2.0 + i * 0.25)
            + np.sin(x * 0.7 + i) * 3
        )

        ax.plot(
            x,
            y,
            marker="o",
            markersize=2.6,
            linewidth=1.25,
            color=color,
            label=f"Model {i + 1}",
        )

    ax.set_title(
        "Training / Metric Curves",
        loc="left",
        fontweight="semibold",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Performance")

    clean_axes(ax)

    ax.legend(
        frameon=False,
        ncol=1,
        loc="upper left",
    )

    # ========================================================
    # 2. Grouped bar
    # ========================================================

    ax = axes[0, 1]

    categories = [
        "A",
        "B",
        "C",
        "D",
    ]

    xx = np.arange(
        len(categories)
    )

    width = 0.14

    values = np.array(
        [
            [82, 76, 88, 79],
            [74, 72, 81, 76],
            [68, 70, 77, 72],
            [63, 66, 71, 69],
            [70, 68, 75, 73],
        ]
    )

    for i, color in enumerate(PASTEL_5):

        ax.bar(
            xx
            + (i - 2) * width,
            values[i],
            width=width,
            color=color,
            edgecolor="none",
            label=f"Model {i + 1}",
            zorder=3,
        )

    ax.set_xticks(
        xx,
        categories,
    )

    ax.set_ylabel("Score (%)")

    ax.set_title(
        "Grouped Model Comparison",
        loc="left",
        fontweight="semibold",
    )

    clean_axes(
        ax,
        grid_y=True,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
    )

    # ========================================================
    # 3. Scatter / bubble
    # ========================================================

    ax = axes[0, 2]

    fps = np.array(
        [82, 104, 119, 126, 113]
    )

    score = np.array(
        [70.2, 69.8, 69.1, 67.4, 72.8]
    )

    size = np.array(
        [20, 4.1, 2.7, 3.2, 3.0]
    )

    bubble = (
        35
        + np.sqrt(size) * 20
    )

    for i in range(4):

        ax.scatter(
            fps[i],
            score[i],
            s=bubble[i],
            color=PASTEL_5[2],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    ax.scatter(
        fps[-1],
        score[-1],
        s=bubble[-1] * 1.05,
        color=ACCENT,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )

    names = [
        "RT-DETR",
        "Model-X",
        "YOLOv10n",
        "YOLOv8n",
        "Ours",
    ]

    for x_value, y_value, name in zip(
        fps,
        score,
        names,
    ):

        ax.annotate(
            name,
            (x_value, y_value),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=6.5,
            fontweight=(
                "semibold"
                if name == "Ours"
                else "normal"
            ),
        )

    ax.set_xlabel(
        "Inference Speed (FPS)"
    )

    ax.set_ylabel(
        "mAP@0.5:0.95 (%)"
    )

    ax.set_title(
        "Accuracy–Efficiency",
        loc="left",
        fontweight="semibold",
    )

    clean_axes(ax)

    # ========================================================
    # 4. Violin + box
    # ========================================================

    ax = axes[1, 0]

    distributions = [
        rng.normal(
            loc=55 + i * 2,
            scale=7 - i * 0.5,
            size=100,
        )
        for i in range(5)
    ]

    violin = ax.violinplot(
        distributions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, color in zip(
        violin["bodies"],
        PASTEL_5,
    ):

        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.88)

    box = ax.boxplot(
        distributions,
        widths=0.12,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "color": "#555555",
            "linewidth": 0.9,
        },
        whiskerprops={
            "color": "#666666",
            "linewidth": 0.7,
        },
        capprops={
            "color": "#666666",
            "linewidth": 0.7,
        },
        boxprops={
            "facecolor": "white",
            "edgecolor": "#555555",
            "linewidth": 0.7,
        },
    )

    ax.set_xticks(
        np.arange(1, 6),
        [
            "A",
            "B",
            "C",
            "D",
            "E",
        ],
    )

    ax.set_ylabel("Score")

    ax.set_title(
        "Distribution Comparison",
        loc="left",
        fontweight="semibold",
    )

    clean_axes(ax)

    # ========================================================
    # 5. Heatmap
    # ========================================================

    ax = axes[1, 1]

    matrix = np.array(
        [
            [0.86, 0.82, 0.77, 0.72, 0.68],
            [0.88, 0.84, 0.79, 0.74, 0.70],
            [0.90, 0.87, 0.82, 0.79, 0.74],
            [0.91, 0.89, 0.85, 0.81, 0.77],
            [0.94, 0.92, 0.89, 0.86, 0.82],
        ]
    )

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "soft_heatmap",
        [
            "#B3CDE4",
            "#F7F7F7",
            "#FBB4AE",
        ],
    )

    image = ax.imshow(
        matrix,
        cmap=cmap,
        aspect="auto",
        vmin=0.65,
        vmax=0.95,
    )

    ax.set_xticks(
        np.arange(5),
        [
            "Normal",
            "Low light",
            "Smoke",
            "Blur",
            "Occlusion",
        ],
    )

    ax.set_yticks(
        np.arange(5),
        [
            "Model 1",
            "Model 2",
            "Model 3",
            "Model 4",
            "Ours",
        ],
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="#333333",
            )

    ax.set_title(
        "Robustness Heatmap",
        loc="left",
        fontweight="semibold",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.045,
        pad=0.03,
    )

    colorbar.ax.tick_params(
        labelsize=6,
        length=2,
    )

    colorbar.outline.set_visible(
        False
    )

    # ========================================================
    # 6. Lollipop ranking
    # ========================================================

    ax = axes[1, 2]

    labels = [
        "Module A",
        "Module B",
        "Module C",
        "Module D",
        "Module E",
    ]

    values = np.array(
        [1.8, 1.5, 1.2, 0.8, 0.5]
    )

    ypos = np.arange(
        len(labels)
    )

    for i, (
        y,
        value,
    ) in enumerate(
        zip(ypos, values)
    ):

        color = (
            ACCENT
            if i == 0
            else PASTEL_5[2]
        )

        ax.hlines(
            y,
            0,
            value,
            color=LIGHT_GRAY,
            linewidth=1.4,
            zorder=1,
        )

        ax.scatter(
            value,
            y,
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_yticks(
        ypos,
        labels,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Performance Gain (%)"
    )

    ax.set_title(
        "Ablation Gain Ranking",
        loc="left",
        fontweight="semibold",
    )

    clean_axes(
        ax,
        grid_x=True,
    )

    # ========================================================
    # Export
    # ========================================================

    fig.suptitle(
        "Scientific Figure Style Gallery",
        fontsize=12,
        fontweight="semibold",
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = (
        OUTPUT_DIR
        / "style_gallery.png"
    )

    pdf = (
        OUTPUT_DIR
        / "style_gallery.pdf"
    )

    svg = (
        OUTPUT_DIR
        / "style_gallery.svg"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        svg,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Style gallery generated:"
    )

    print(
        f"PNG: {png}"
    )

    print(
        f"PDF: {pdf}"
    )

    print(
        f"SVG: {svg}"
    )


if __name__ == "__main__":
    main()