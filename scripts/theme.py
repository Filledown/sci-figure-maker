from pathlib import Path
from typing import Any

import yaml


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PALETTE_FILE = PROJECT_ROOT / "palettes" / "palettes.yaml"
DEFAULT_PRESET_FILE = PROJECT_ROOT / "presets" / "default.yaml"


# ------------------------------------------------------------
# YAML loading
# ------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.

    ValueError
        If the YAML file is empty or does not contain
        a dictionary-like structure.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}, "
            f"but received {type(data).__name__}."
        )

    return data


# ------------------------------------------------------------
# Palette configuration
# ------------------------------------------------------------

def load_palettes() -> dict[str, Any]:
    """
    Load the complete scientific palette configuration.
    """
    return load_yaml(PALETTE_FILE)


def get_palette(name: str) -> list[str]:
    """
    Return a standard palette as a list of HEX colors.

    Example
    -------
    get_palette("recognition_models")
    """

    config = load_palettes()

    palettes = config.get("palettes", {})

    if name not in palettes:
        available = ", ".join(sorted(palettes.keys()))

        raise KeyError(
            f"Unknown palette '{name}'. "
            f"Available palettes: {available}"
        )

    palette = palettes[name]

    colors = palette.get("colors")

    if not colors:
        raise ValueError(
            f"Palette '{name}' does not define a standard "
            f"'colors' list."
        )

    return colors


def get_semantic_palette(name: str) -> dict[str, Any]:
    """
    Return a semantic palette.

    Semantic palettes may contain roles such as:

    - baseline
    - highlight

    rather than one simple color list.
    """

    config = load_palettes()

    palettes = config.get("palettes", {})

    if name not in palettes:
        available = ", ".join(sorted(palettes.keys()))

        raise KeyError(
            f"Unknown palette '{name}'. "
            f"Available palettes: {available}"
        )

    return palettes[name]


def get_default_palette(purpose: str) -> str:
    """
    Return the palette name assigned to a specific purpose.

    Examples
    --------
    get_default_palette("categorical")
    get_default_palette("model_comparison")
    """

    config = load_palettes()

    defaults = config.get("defaults", {})

    if purpose not in defaults:
        available = ", ".join(sorted(defaults.keys()))

        raise KeyError(
            f"Unknown palette purpose '{purpose}'. "
            f"Available purposes: {available}"
        )

    return defaults[purpose]


# ------------------------------------------------------------
# Figure preset configuration
# ------------------------------------------------------------

def load_default_preset() -> dict[str, Any]:
    """
    Load the default scientific figure preset.
    """
    return load_yaml(DEFAULT_PRESET_FILE)


def get_figure_width(preset_name: str = "double_column") -> float:
    """
    Return publication figure width in inches.

    Examples
    --------
    get_figure_width("single_column")
    get_figure_width("double_column")
    """

    preset = load_default_preset()

    width_presets = (
        preset
        .get("figure", {})
        .get("width_presets", {})
    )

    if preset_name not in width_presets:
        available = ", ".join(sorted(width_presets.keys()))

        raise KeyError(
            f"Unknown width preset '{preset_name}'. "
            f"Available presets: {available}"
        )

    return float(
        width_presets[preset_name]["width_in"]
    )


def get_export_dpi() -> int:
    """
    Return the default PNG export DPI.
    """

    preset = load_default_preset()

    return int(
        preset["export"]["png"]["dpi"]
    )


# ------------------------------------------------------------
# Diagnostic utility
# ------------------------------------------------------------

def describe_theme() -> None:
    """
    Print a short summary of the currently loaded theme.
    Useful for debugging and setup verification.
    """

    palettes = load_palettes()
    preset = load_default_preset()

    palette_names = list(
        palettes.get("palettes", {}).keys()
    )

    print("sci-figure-maker theme")
    print("------------------------")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Palette file: {PALETTE_FILE}")
    print(f"Preset file: {DEFAULT_PRESET_FILE}")
    print()
    print(f"Available palettes: {len(palette_names)}")

    for name in palette_names:
        print(f"  - {name}")

    print()
    print(
        "Single-column width:",
        get_figure_width("single_column"),
        "in"
    )

    print(
        "Double-column width:",
        get_figure_width("double_column"),
        "in"
    )

    print(
        "PNG export DPI:",
        get_export_dpi()
    )


if __name__ == "__main__":
    describe_theme()