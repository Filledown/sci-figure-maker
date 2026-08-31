from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.inspect_data import inspect_data
from scripts.load_data import load_data
from scripts.select_plot import build_selection_report


QUALITY_PRIORITY = [
    "map50_95",
    "map50",
    "f1",
    "accuracy",
    "miou",
    "dice",
    "auc",
    "precision",
    "recall",
]

EFFICIENCY_PRIORITY = [
    "fps",
    "latency",
]

COMPLEXITY_PRIORITY = [
    "params",
    "flops",
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _normalize(
    value: str,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def _role_column(
    inspection: dict[str, Any],
    role: str,
) -> str | None:

    info = inspection.get(
        "semantic_roles",
        {},
    ).get(
        role
    )

    if not info:
        return None

    return info.get(
        "column"
    )


def _metric_column(
    inspection: dict[str, Any],
    names: list[str],
) -> str | None:

    metrics = inspection.get(
        "metrics",
        {},
    )

    for name in names:

        if name in metrics:

            return metrics[
                name
            ]["column"]

    return None


def _quality_columns(
    inspection: dict[str, Any],
) -> list[str]:

    metrics = inspection.get(
        "metrics",
        {},
    )

    columns: list[str] = []

    for name in QUALITY_PRIORITY:

        info = metrics.get(
            name
        )

        if (
            info
            and info.get(
                "family"
            )
            == "quality"
        ):

            column = info[
                "column"
            ]

            if column not in columns:

                columns.append(
                    column
                )

    return columns


def _curve_column(
    inspection: dict[str, Any],
    name: str,
) -> str | None:

    info = inspection.get(
        "curve_variables",
        {},
    ).get(
        name
    )

    if not info:
        return None

    return info.get(
        "column"
    )


# ------------------------------------------------------------
# Training semantic planning
# ------------------------------------------------------------

def _training_families(
    data: pd.DataFrame,
    epoch_column: str,
) -> dict[str, list[str]]:

    loss_columns: list[str] = []

    performance_columns: list[str] = []

    learning_rate_columns: list[
        str
    ] = []

    for column in data.columns:

        if column == epoch_column:
            continue

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            continue

        name = _normalize(
            column
        )

        if "loss" in name:

            loss_columns.append(
                column
            )

            continue

        if (
            name
            in {
                "lr",
                "learningrate",
                "learningrate0",
                "learningrate1",
                "learningrate2",
            }
            or name.startswith(
                "lr"
            )
        ):

            learning_rate_columns.append(
                column
            )

            continue

        if (
            name.startswith(
                "map"
            )
            or name.startswith(
                "precision"
            )
            or name.startswith(
                "recall"
            )
            or name.startswith(
                "accuracy"
            )
            or name.startswith(
                "acc"
            )
            or name.startswith(
                "f1"
            )
            or name.startswith(
                "miou"
            )
            or name.startswith(
                "iou"
            )
            or name.startswith(
                "dice"
            )
            or name.startswith(
                "auc"
            )
        ):

            performance_columns.append(
                column
            )

    return {
        "loss": loss_columns,
        "performance": (
            performance_columns
        ),
        "learning_rate": (
            learning_rate_columns
        ),
    }


def _training_summary(
    data: pd.DataFrame,
    epoch_column: str,
    families: dict[
        str,
        list[str],
    ],
) -> dict[str, Any]:

    summary: dict[
        str,
        Any,
    ] = {}

    validation_losses = [
        column
        for column
        in families["loss"]
        if (
            "val"
            in _normalize(
                column
            )
            or "validation"
            in _normalize(
                column
            )
        )
    ]

    if validation_losses:

        column = validation_losses[
            0
        ]

        numeric = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if numeric.notna().any():

            index = numeric.idxmin()

            summary[
                "best_validation_loss"
            ] = {
                "column": column,
                "epoch": data.loc[
                    index,
                    epoch_column,
                ],
                "value": float(
                    numeric.loc[
                        index
                    ]
                ),
            }

    best_performance: dict[
        str,
        Any,
    ] = {}

    for column in families[
        "performance"
    ]:

        numeric = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if not numeric.notna().any():
            continue

        index = numeric.idxmax()

        best_performance[
            column
        ] = {
            "epoch": data.loc[
                index,
                epoch_column,
            ],
            "value": float(
                numeric.loc[
                    index
                ]
            ),
        }

    if best_performance:

        summary[
            "best_performance"
        ] = best_performance

    return summary


def _training_plans(
    data: pd.DataFrame,
    inspection: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:

    epoch_column = _role_column(
        inspection,
        "epoch",
    )

    if not epoch_column:

        return [], {}

    families = _training_families(
        data,
        epoch_column,
    )

    plans: list[
        dict[str, Any]
    ] = []

    if families[
        "loss"
    ]:

        plans.append(
            {
                "figure": (
                    "training_loss"
                ),
                "title": (
                    "Training Loss"
                ),
                "template": "line",
                "priority": 100,
                "reason": (
                    "Loss metrics were separated "
                    "from performance metrics."
                ),
                "parameters": {
                    "x": epoch_column,
                    "ys": families[
                        "loss"
                    ],
                    "labels": families[
                        "loss"
                    ],
                    "title": (
                        "Training Loss"
                    ),
                    "xlabel": "Epoch",
                    "ylabel": "Loss",
                },
            }
        )

    if families[
        "performance"
    ]:

        plans.append(
            {
                "figure": (
                    "training_metrics"
                ),
                "title": (
                    "Training Performance"
                ),
                "template": "line",
                "priority": 95,
                "reason": (
                    "Performance metrics were "
                    "separated from loss."
                ),
                "parameters": {
                    "x": epoch_column,
                    "ys": families[
                        "performance"
                    ],
                    "labels": families[
                        "performance"
                    ],
                    "title": (
                        "Training Performance"
                    ),
                    "xlabel": "Epoch",
                    "ylabel": (
                        "Performance"
                    ),
                },
            }
        )

    if families[
        "learning_rate"
    ]:

        plans.append(
            {
                "figure": (
                    "learning_rate_curve"
                ),
                "title": (
                    "Learning Rate"
                ),
                "template": "line",
                "priority": 80,
                "reason": (
                    "Learning-rate values "
                    "use a separate scale."
                ),
                "parameters": {
                    "x": epoch_column,
                    "ys": families[
                        "learning_rate"
                    ],
                    "labels": families[
                        "learning_rate"
                    ],
                    "title": (
                        "Learning Rate"
                    ),
                    "xlabel": "Epoch",
                    "ylabel": (
                        "Learning Rate"
                    ),
                },
            }
        )

    summary = _training_summary(
        data,
        epoch_column,
        families,
    )

    return (
        plans,
        summary,
    )


# ------------------------------------------------------------
# Confidence / threshold curves
# ------------------------------------------------------------

def _confidence_plans(
    inspection: dict[str, Any],
) -> list[dict[str, Any]]:

    confidence = _curve_column(
        inspection,
        "confidence",
    )

    if not confidence:

        return []

    metrics = inspection.get(
        "metrics",
        {},
    )

    definitions = [
        (
            "precision",
            "precision_confidence",
            "Precision–Confidence Curve",
            "Precision",
        ),
        (
            "recall",
            "recall_confidence",
            "Recall–Confidence Curve",
            "Recall",
        ),
        (
            "f1",
            "f1_confidence",
            "F1–Confidence Curve",
            "F1",
        ),
    ]

    plans: list[
        dict[str, Any]
    ] = []

    for (
        metric_name,
        figure_id,
        title,
        ylabel,
    ) in definitions:

        info = metrics.get(
            metric_name
        )

        if not info:
            continue

        y_column = info.get(
            "column"
        )

        if not y_column:
            continue

        plans.append(
            {
                "figure": figure_id,
                "title": title,
                "template": "line",
                "priority": 100,
                "reason": (
                    f"{ylabel} values were "
                    "measured across confidence "
                    "thresholds."
                ),
                "parameters": {
                    "x": confidence,
                    "ys": [
                        y_column
                    ],
                    "labels": [
                        ylabel
                    ],
                    "title": title,
                    "xlabel": (
                        "Confidence"
                    ),
                    "ylabel": ylabel,
                    "xlim": (
                        0.0,
                        1.0,
                    ),
                    "ylim": (
                        0.0,
                        1.0,
                    ),
                },
            }
        )

    return plans


# ------------------------------------------------------------
# Standard figure plans
# ------------------------------------------------------------

def _plan_for_figure(
    inspection: dict[str, Any],
    recommendation: dict[str, Any],
    *,
    highlight: str | None,
) -> dict[str, Any] | None:

    figure = recommendation[
        "figure"
    ]

    model_column = _role_column(
        inspection,
        "model",
    )

    quality_columns = (
        _quality_columns(
            inspection
        )
    )

    quality_column = (
        quality_columns[0]
        if quality_columns
        else None
    )

    # --------------------------------------------------------
    # PR
    # --------------------------------------------------------

    if figure == "pr_curve":

        recall = _curve_column(
            inspection,
            "recall_axis",
        )

        precision = _curve_column(
            inspection,
            "precision_axis",
        )

        if (
            not recall
            or not precision
        ):

            return None

        return {
            "figure": "pr_curve",
            "title": (
                "Precision–Recall Curve"
            ),
            "template": "line",
            "priority": (
                recommendation[
                    "priority"
                ]
            ),
            "reason": (
                recommendation[
                    "reason"
                ]
            ),
            "parameters": {
                "x": recall,
                "y": precision,
                "group": (
                    model_column
                ),
                "highlight": (
                    highlight
                ),
                "title": (
                    "Precision–Recall Curve"
                ),
                "xlabel": "Recall",
                "ylabel": "Precision",
            },
        }

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    if figure == "roc_curve":

        fpr = _curve_column(
            inspection,
            "false_positive_rate",
        )

        tpr = _curve_column(
            inspection,
            "true_positive_rate",
        )

        if (
            not fpr
            or not tpr
        ):

            return None

        return {
            "figure": "roc_curve",
            "title": "ROC Curve",
            "template": "line",
            "priority": (
                recommendation[
                    "priority"
                ]
            ),
            "reason": (
                recommendation[
                    "reason"
                ]
            ),
            "parameters": {
                "x": fpr,
                "y": tpr,
                "group": (
                    model_column
                ),
                "highlight": (
                    highlight
                ),
                "title": "ROC Curve",
                "xlabel": (
                    "False Positive Rate"
                ),
                "ylabel": (
                    "True Positive Rate"
                ),
            },
        }

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    if (
        figure
        == "model_comparison"
    ):

        if (
            not model_column
            or not quality_columns
        ):

            return None

        return {
            "figure": (
                "model_comparison"
            ),
            "title": (
                "Model Comparison"
            ),
            "template": (
                "grouped_bar"
            ),
            "priority": (
                recommendation[
                    "priority"
                ]
            ),
            "reason": (
                recommendation[
                    "reason"
                ]
            ),
            "parameters": {
                "category": (
                    model_column
                ),
                "values": (
                    quality_columns[
                        :4
                    ]
                ),
                "title": (
                    "Model Comparison"
                ),
                "xlabel": None,
                "ylabel": (
                    "Performance"
                ),
            },
        }

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    if (
        figure
        == "model_ranking"
    ):

        if (
            not model_column
            or not quality_column
        ):

            return None

        return {
            "figure": (
                "model_ranking"
            ),
            "title": (
                "Model Ranking"
            ),
            "template": "lollipop",
            "priority": (
                recommendation[
                    "priority"
                ]
            ),
            "reason": (
                recommendation[
                    "reason"
                ]
            ),
            "parameters": {
                "category": (
                    model_column
                ),
                "value": (
                    quality_column
                ),
                "highlight": (
                    highlight
                ),
                "sort": True,
                "title": (
                    "Model Ranking"
                ),
                "xlabel": (
                    quality_column
                ),
            },
        }

    # --------------------------------------------------------
    # Accuracy-efficiency
    # --------------------------------------------------------

    if (
        figure
        == "accuracy_efficiency"
    ):

        efficiency = (
            _metric_column(
                inspection,
                EFFICIENCY_PRIORITY,
            )
        )

        complexity = (
            _metric_column(
                inspection,
                COMPLEXITY_PRIORITY,
            )
        )

        if (
            not model_column
            or not quality_column
            or not efficiency
        ):

            return None

        return {
            "figure": (
                "accuracy_efficiency"
            ),
            "title": (
                "Accuracy–Efficiency Trade-off"
            ),
            "template": (
                "scatter_bubble"
            ),
            "priority": (
                recommendation[
                    "priority"
                ]
            ),
            "reason": (
                recommendation[
                    "reason"
                ]
            ),
            "parameters": {
                "x": efficiency,
                "y": (
                    quality_column
                ),
                "label": (
                    model_column
                ),
                "size": (
                    complexity
                ),
                "highlight": (
                    highlight
                ),
                "title": (
                    "Accuracy–Efficiency"
                ),
                "xlabel": (
                    efficiency
                ),
                "ylabel": (
                    quality_column
                ),
            },
        }

    # --------------------------------------------------------
    # Ablation
    # --------------------------------------------------------

    if figure == "ablation":

        variant = _role_column(
            inspection,
            "variant",
        )

        if (
            not variant
            or not quality_columns
        ):

            return None

        return {
            "figure": "ablation",
            "title": (
                "Ablation Study"
            ),
            "template": (
                "grouped_bar"
            ),
            "priority": (
                recommendation[
                    "priority"
                ]
            ),
            "reason": (
                recommendation[
                    "reason"
                ]
            ),
            "parameters": {
                "category": variant,
                "values": (
                    quality_columns[
                        :4
                    ]
                ),
                "title": (
                    "Ablation Study"
                ),
                "xlabel": None,
                "ylabel": (
                    "Performance"
                ),
            },
        }

    # --------------------------------------------------------
    # Robustness
    # --------------------------------------------------------

    if (
        figure
        == "robustness_heatmap"
    ):

        scenario = _role_column(
            inspection,
            "scenario",
        )

        if (
            not model_column
            or not scenario
            or not quality_column
        ):

            return None

        return {
            "figure": (
                "robustness_heatmap"
            ),
            "title": (
                "Robustness Heatmap"
            ),
            "template": "heatmap",
            "priority": (
                recommendation[
                    "priority"
                ]
            ),
            "reason": (
                recommendation[
                    "reason"
                ]
            ),
            "parameters": {
                "row": (
                    model_column
                ),
                "column": scenario,
                "value": (
                    quality_column
                ),
                "title": (
                    "Robustness Heatmap"
                ),
            },
        }

    return None


# ------------------------------------------------------------
# Full planner
# ------------------------------------------------------------

def build_figure_plan(
    data: pd.DataFrame,
    inspection: dict[str, Any],
    selection: dict[str, Any],
    *,
    highlight: str | None = None,
    top: int = 3,
) -> dict[str, Any]:

    plans: list[
        dict[str, Any]
    ] = []

    training_summary: dict[
        str,
        Any,
    ] = {}

    for recommendation in (
        selection.get(
            "recommendations",
            [],
        )
    ):

        # ----------------------------------------------------
        # Training expands into multiple semantic figures
        # ----------------------------------------------------

        if (
            recommendation[
                "figure"
            ]
            == "training_curve"
        ):

            (
                training_plans,
                summary,
            ) = _training_plans(
                data,
                inspection,
            )

            plans.extend(
                training_plans
            )

            if summary:

                training_summary = (
                    summary
                )

            continue

        # ----------------------------------------------------
        # Confidence expands into P/R/F1 curves
        # ----------------------------------------------------

        if (
            recommendation[
                "figure"
            ]
            == "confidence_curve"
        ):

            plans.extend(
                _confidence_plans(
                    inspection
                )
            )

            continue

        # ----------------------------------------------------
        # Normal one-to-one plans
        # ----------------------------------------------------

        plan = _plan_for_figure(
            inspection,
            recommendation,
            highlight=highlight,
        )

        if plan is not None:

            plans.append(
                plan
            )

    # --------------------------------------------------------
    # Remove duplicate figure IDs
    # --------------------------------------------------------

    unique_plans: list[
        dict[str, Any]
    ] = []

    seen: set[str] = set()

    for plan in plans:

        figure = plan[
            "figure"
        ]

        if figure in seen:
            continue

        seen.add(
            figure
        )

        unique_plans.append(
            plan
        )

    if top > 0:

        unique_plans = (
            unique_plans[
                :top
            ]
        )

    return {
        "source": inspection.get(
            "source"
        ),
        "plans": unique_plans,
        "training_summary": (
            training_summary
        ),
        "warnings": inspection.get(
            "warnings",
            [],
        ),
    }


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_sheet(
    value: str,
) -> str | int:

    try:
        return int(value)

    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Build concrete scientific "
            "figure plans from experimental data."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--sheet",
        default="0",
    )

    parser.add_argument(
        "--highlight",
        default=None,
    )

    parser.add_argument(
        "--top",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/figure_plan/"
            "figure_plan.json"
        ),
    )

    return parser


def main() -> None:

    args = (
        build_parser()
        .parse_args()
    )

    input_path = Path(
        args.input
    )

    data = load_data(
        input_path,
        sheet_name=parse_sheet(
            str(
                args.sheet
            )
        ),
    )

    inspection = inspect_data(
        data,
        source=input_path,
    )

    selection = (
        build_selection_report(
            inspection
        )
    )

    plan = build_figure_plan(
        data=data,
        inspection=inspection,
        selection=selection,
        highlight=args.highlight,
        top=args.top,
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            plan,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(
        "sci-figure-maker figure planner"
    )

    print(
        "==============================="
    )

    if not plan[
        "plans"
    ]:

        print(
            "No renderable figure "
            "plans were created."
        )

    else:

        for index, item in enumerate(
            plan[
                "plans"
            ],
            start=1,
        ):

            print()

            print(
                f"{index}. "
                f"{item['title']}"
            )

            print(
                f"   Figure: "
                f"{item['figure']}"
            )

            print(
                f"   Template: "
                f"{item['template']}"
            )

    print()

    print(
        f"Plan saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()