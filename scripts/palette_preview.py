from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scripts.theme import load_palettes


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "assets"
OUTPUT_FILE = OUTPUT_DIR / "palette-preview.png"


def extract_colors(palette: dict) -> list[str]:
    """
    Convert a palette definition into a simple list of colors.

    Standard palettes contain:
        colors: [...]

    Semantic palettes may instead contain:
        baseline: [...]
        highlight: "#..."
    """

    if "colors" in palette:
        return list(palette["colors"])

    colors = []

    baseline = palette.get("baseline", [])
    if isinstance(baseline, list):
        colors.extend(baseline)

    highlight = palette.get("highlight")
    if highlight:
        colors.append(highlight)

    return colors


def create_palette_preview() -> Path:
    """
    Generate a preview image containing all configured palettes.
    """

    config = load_palettes()
    palettes = config.get("palettes", {})

    if not palettes:
        raise ValueError("No palettes were found in palettes.yaml.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    palette_items = list(palettes.items())

    row_height = 0.85
    figure_height = max(3.0, len(palette_items) * row_height)

    fig, ax = plt.subplots(
        figsize=(10, figure_height)
    )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(palette_items))
    ax.axis("off")

    for row_index, (name, palette) in enumerate(palette_items):

        y = len(palette_items) - row_index - 1

        colors = extract_colors(palette)

        # Palette name
        ax.text(
            0.1,
            y + 0.5,
            name,
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

        if not colors:
            ax.text(
                2.7,
                y + 0.5,
                "No displayable colors",
                va="center",
                fontsize=9,
            )
            continue

        start_x = 2.7
        available_width = 6.8

        swatch_width = available_width / len(colors)

        for index, color in enumerate(colors):

            x = start_x + index * swatch_width

            rectangle = Rectangle(
                (x, y + 0.15),
                swatch_width,
                0.7,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
            )

            ax.add_patch(rectangle)

    ax.set_title(
        "sci-figure-maker — Scientific Palette Preview",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return OUTPUT_FILE


if __name__ == "__main__":

    output_path = create_palette_preview()

    print("Palette preview generated successfully.")
    print(f"Output: {output_path}")