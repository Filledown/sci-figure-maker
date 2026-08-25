from __future__ import annotations

from typing import Callable

from templates import (
    render_confusion_matrix,
    render_grouped_bar,
    render_heatmap,
    render_line,
    render_lollipop,
    render_ridge,
    render_sankey,
    render_scatter_bubble,
    render_stacked_area,
    render_violin_box,
)


TEMPLATE_REGISTRY: dict[str, Callable] = {
    "line": render_line,
    "grouped_bar": render_grouped_bar,
    "scatter_bubble": render_scatter_bubble,
    "violin_box": render_violin_box,
    "heatmap": render_heatmap,
    "lollipop": render_lollipop,
    "stacked_area": render_stacked_area,
    "ridge": render_ridge,
    "confusion_matrix": render_confusion_matrix,
    "sankey": render_sankey,
}


SEMANTIC_TO_TEMPLATE = {
    "training_curve": "line",
    "metric_curve": "line",
    "pr_curve": "line",
    "roc_curve": "line",
    "threshold_curve": "line",

    "model_comparison": "grouped_bar",

    "accuracy_efficiency": "scatter_bubble",
    "tradeoff": "scatter_bubble",

    "distribution_comparison": "violin_box",
    "stability_distribution": "violin_box",

    "robustness_heatmap": "heatmap",
    "scenario_matrix": "heatmap",

    "model_ranking": "lollipop",
    "ablation": "lollipop",

    "category_proportion": "stacked_area",
    "error_composition": "stacked_area",

    "score_distribution": "ridge",
    "confidence_distribution": "ridge",
    "iou_distribution": "ridge",

    "confusion_matrix": "confusion_matrix",

    "error_flow": "sankey",
    "class_transition": "sankey",
}
