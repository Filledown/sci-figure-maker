from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import pandas as pd

from scripts.figure_style import apply_publication_style, get_palette
from templates.base import require_columns, save_figure


def _layout(names, totals, bottom=0.08, top=0.92, gap=0.035):
    usable = top - bottom - gap * max(len(names) - 1, 0)
    scale = usable / max(sum(totals.get(name, 0) for name in names), 1e-12)
    positions = {}
    y = top
    for name in names:
        height = totals.get(name, 0) * scale
        positions[name] = (y - height, y)
        y = y - height - gap
    return positions, scale


def _flow_patch(x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.40):
    c = (x1 - x0) * 0.42
    verts = [
        (x0, y0a),
        (x0 + c, y0a), (x1 - c, y1a), (x1, y1a),
        (x1, y1b),
        (x1 - c, y1b), (x0 + c, y0b), (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)


def render_sankey(
    data: pd.DataFrame,
    source: str,
    target: str,
    value: str,
    *,
    title: str | None = None,
    output_dir: str | Path = "outputs/sankey",
) -> dict[str, Path]:
    """Two-stage Sankey/alluvial diagram for error flow and class transitions."""
    require_columns(data, [source, target, value])
    apply_publication_style()

    flows = data[[source, target, value]].copy()
    flows[value] = pd.to_numeric(flows[value], errors="coerce")
    flows = flows.dropna()
    flows = flows[flows[value] > 0]
    if flows.empty:
        raise ValueError("Sankey requires positive flow values.")

    sources = [str(x) for x in pd.unique(flows[source])]
    targets = [str(x) for x in pd.unique(flows[target])]

    source_totals = defaultdict(float)
    target_totals = defaultdict(float)
    for _, row in flows.iterrows():
        source_totals[str(row[source])] += float(row[value])
        target_totals[str(row[target])] += float(row[value])

    source_pos, source_scale = _layout(sources, source_totals)
    target_pos, target_scale = _layout(targets, target_totals)
    scale = min(source_scale, target_scale)

    # Re-layout both sides with one common scale for visually faithful flow widths.
    def layout_fixed(names, totals):
        gap = 0.035
        top = 0.92
        positions = {}
        y = top
        for name in names:
            height = totals[name] * scale
            positions[name] = (y - height, y)
            y = y - height - gap
        return positions

    source_pos = layout_fixed(sources, source_totals)
    target_pos = layout_fixed(targets, target_totals)

    fig, ax = plt.subplots(figsize=(4.1, 2.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    source_colors = {name: color for name, color in zip(sources, get_palette(len(sources)))}
    target_colors = {name: color for name, color in zip(targets, get_palette(len(targets)))}

    bar_w = 0.025
    x0, x1 = 0.14, 0.86

    for name in sources:
        y0, y1 = source_pos[name]
        ax.add_patch(Rectangle((x0 - bar_w, y0), bar_w, y1 - y0, facecolor=source_colors[name], edgecolor="none", alpha=0.92))
        ax.text(x0 - bar_w - 0.015, (y0 + y1) / 2, name, ha="right", va="center", fontsize=6.5)

    for name in targets:
        y0, y1 = target_pos[name]
        ax.add_patch(Rectangle((x1, y0), bar_w, y1 - y0, facecolor=target_colors[name], edgecolor="none", alpha=0.92))
        ax.text(x1 + bar_w + 0.015, (y0 + y1) / 2, name, ha="left", va="center", fontsize=6.5)

    source_cursor = {name: source_pos[name][1] for name in sources}
    target_cursor = {name: target_pos[name][1] for name in targets}

    for _, row in flows.iterrows():
        s = str(row[source])
        t = str(row[target])
        h = float(row[value]) * scale

        sy1 = source_cursor[s]
        sy0 = sy1 - h
        source_cursor[s] = sy0

        ty1 = target_cursor[t]
        ty0 = ty1 - h
        target_cursor[t] = ty0

        ax.add_patch(_flow_patch(
            x0, x1,
            sy0, sy1,
            ty0, ty1,
            source_colors[s],
            alpha=0.33,
        ))

    if title:
        ax.set_title(title, loc="left", fontweight="semibold")

    fig.tight_layout()
    return save_figure(fig, output_dir)
