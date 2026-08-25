from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.inspect_data import inspect_data
from scripts.load_data import load_data
from scripts.select_plot import build_selection_report
from scripts.template_registry import SEMANTIC_TO_TEMPLATE


# ------------------------------------------------------------
# Metric priorities
# ------------------------------------------------------------

QUALITY_PRIORITY = [
    "map50_95",
    "map50",
    "f1",
    "accuracy",
    "miou",
    "dice",
    "auc",
    "ap",
    "precision",
    "recall",
    "iou",
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
# Small lookup helpers
# ------------------------------------------------------------

def role_column(
    inspection: dict[str, Any],
    role: str,
) -> str | None:

    role_info = inspection.get(
        "semantic_roles",
        {},
    ).get(role)

    if not role_info:
        return None

    return role_info.get("column")


def metric_column(
    inspection: dict[str, Any],
    candidates: list[str],
) -> str | None:

    metrics = inspection.get(
        "metrics",
        {},
    )

    for metric_name in candidates:

        metric_info = metrics.get(
            metric_name
        )

        if metric_info:
            return metric_info.get(
                "column"
            )

    return None


def metric_columns_by_family(
    inspection: dict[str, Any],
    family: str,
) -> list[str]:

    result = []

    for metric_info in inspection.get(
        "metrics",
        {},
    ).values():

        if (
            metric_info.get("family")
            == family
        ):
            column = metric_info.get(
                "column"
            )

            if (
                column
                and column not in result
            ):
                result.append(column)

    return result


def curve_column(
    inspection: dict[str, Any],
    variable: str,
) -> str | None:

    info = inspection.get(
        "curve_variables",
        {},
    ).get(variable)

    if not info:
        return None

    return info.get("column")


# ------------------------------------------------------------
# Highlight detection
# ------------------------------------------------------------

def infer_highlight(
    data: pd.DataFrame,
    identity_column: str | None,
    requested: str | None = None,
) -> str | None:
    """
    Prefer an explicitly supplied highlight.

    Otherwise only recognize conservative,
    common labels such as 'Ours'.
    """

    if requested:
        return requested

    if not identity_column:
        return None

    if identity_column not in data.columns:
        return None

    values = [
        str(value)
        for value in data[
            identity_column
        ].dropna().unique()
    ]

    preferred = {
        "ours",
        "our model",
        "proposed",
        "proposed model",
    }

    for value in values:

        if value.strip().lower() in preferred:
            return value

    return None


# ------------------------------------------------------------
# Training-curve helper
# ------------------------------------------------------------

def find_training_columns(
    data: pd.DataFrame,
    x_column: str | None,
) -> list[str]:
    """
    Find plausible training/metric columns.

    This intentionally uses column-name hints as well
    as numeric dtype because one table may contain
    train_loss, val_loss, mAP, Precision, Recall, etc.
    """

    tokens = (
        "loss",
        "acc",
        "accuracy",
        "precision",
        "recall",
        "map",
        "f1",
        "auc",
        "iou",
        "dice",
        "lr",
        "learning_rate",
    )

    result = []

    for column in data.columns:

        if column == x_column:
            continue

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            continue

        normalized = (
            str(column)
            .strip()
            .lower()
        )

        if any(
            token in normalized
            for token in tokens
        ):
            result.append(column)

    return result

# ------------------------------------------------------------
# Training figure expansion
# ------------------------------------------------------------

def pretty_metric_label(
    column: str,
) -> str:
    """
    Convert common raw column names into
    cleaner publication labels.
    """

    normalized = (
        str(column)
        .strip()
        .lower()
    )

    labels = {
        "train_loss": "Train loss",
        "training_loss": "Train loss",
        "val_loss": "Validation loss",
        "validation_loss": "Validation loss",
        "map50": "mAP@0.5",
        "map50_95": "mAP@0.5:0.95",
        "precision": "Precision",
        "recall": "Recall",
        "accuracy": "Accuracy",
        "acc": "Accuracy",
        "f1": "F1",
        "f1_score": "F1",
        "auc": "AUC",
        "iou": "IoU",
        "miou": "mIoU",
        "dice": "Dice",
    }

    return labels.get(
        normalized,
        str(column).replace(
            "_",
            " ",
        ),
    )


def plan_training_variants(
    data: pd.DataFrame,
    inspection: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Split training data into scientifically
    compatible line figures.

    Loss metrics and performance metrics should
    not normally share one y-axis because their
    scales and meanings are different.
    """

    epoch = role_column(
        inspection,
        "epoch",
    )

    if not epoch:
        return []

    loss_columns = []
    performance_columns = []

    performance_tokens = (
        "map",
        "accuracy",
        "acc",
        "precision",
        "recall",
        "f1",
        "auc",
        "iou",
        "miou",
        "dice",
    )

    for column in data.columns:

        if column == epoch:
            continue

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            continue

        normalized = (
            str(column)
            .strip()
            .lower()
        )

        if "loss" in normalized:

            loss_columns.append(
                column
            )

            continue

        if any(
            token in normalized
            for token
            in performance_tokens
        ):

            performance_columns.append(
                column
            )

    variants = []


    if loss_columns:

        variants.append(
            {
                "figure": (
                    "training_loss_curve"
                ),
                "title": (
                    "Training Loss Curves"
                ),
                "parameters": {
                    "x": epoch,
                    "ys": (
                        loss_columns[:4]
                    ),
                    "labels": [
                        pretty_metric_label(
                            column
                        )
                        for column
                        in loss_columns[:4]
                    ],
                    "title": (
                        "Training Loss Curves"
                    ),
                    "xlabel": "Epoch",
                    "ylabel": "Loss",
                },
            }
        )

    # --------------------------------------------------------
    # Performance curves
    # --------------------------------------------------------

    if performance_columns:

        variants.append(
            {
                "figure": (
                    "training_metric_curve"
                ),
                "title": (
                    "Training Metric Curves"
                ),
                "parameters": {
                    "x": epoch,
                    "ys": (
                        performance_columns[
                            :6
                        ]
                    ),
                    "labels": [
                        pretty_metric_label(
                            column
                        )
                        for column
                        in performance_columns[
                            :6
                        ]
                    ],
                    "title": (
                        "Training Metric Curves"
                    ),
                    "xlabel": "Epoch",
                    "ylabel": (
                        "Metric value"
                    ),
                },
            }
        )

    return variants

def plan_parameters(
    figure_id: str,
    data: pd.DataFrame,
    inspection: dict[str, Any],
    highlight: str | None = None,
) -> dict[str, Any] | None:

    model = role_column(
        inspection,
        "model",
    )

    variant = role_column(
        inspection,
        "variant",
    )

    scenario = role_column(
        inspection,
        "scenario",
    )

    epoch = role_column(
        inspection,
        "epoch",
    )

    quality = metric_column(
        inspection,
        QUALITY_PRIORITY,
    )

    efficiency = metric_column(
        inspection,
        EFFICIENCY_PRIORITY,
    )

    complexity = metric_column(
        inspection,
        COMPLEXITY_PRIORITY,
    )

    detected_highlight = (
        infer_highlight(
            data,
            model,
            highlight,
        )
    )

    # --------------------------------------------------------
    # Accuracy–Efficiency
    # --------------------------------------------------------

    if figure_id == "accuracy_efficiency":

        if not (
            model
            and quality
            and efficiency
        ):
            return None

        return {
            "x": efficiency,
            "y": quality,
            "label": model,
            "size": complexity,
            "highlight": detected_highlight,
            "xlabel": efficiency,
            "ylabel": quality,
            "title": "Accuracy–Efficiency",
        }

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    if figure_id == "model_comparison":

        if not model:
            return None

        quality_columns = (
            metric_columns_by_family(
                inspection,
                "quality",
            )
        )

        if not quality_columns:
            return None

        return {
            "category": model,
            "values": quality_columns[:4],
            "title": "Model Comparison",
            "ylabel": "Performance",
        }

    # --------------------------------------------------------
    # Model ranking
    # --------------------------------------------------------

    if figure_id == "model_ranking":

        if not (
            model
            and quality
        ):
            return None

        return {
            "category": model,
            "value": quality,
            "highlight": detected_highlight,
            "sort": True,
            "title": "Model Ranking",
            "xlabel": quality,
        }

    # --------------------------------------------------------
    # Training curve
    # --------------------------------------------------------

    if figure_id == "training_curve":

        if not epoch:
            return None

        y_columns = find_training_columns(
            data,
            epoch,
        )

        if not y_columns:
            return None

        return {
            "x": epoch,
            "ys": y_columns[:6],
            "title": "Training / Metric Curves",
            "xlabel": epoch,
        }

    # --------------------------------------------------------
    # Ablation
    # --------------------------------------------------------

    if figure_id == "ablation":

        if not (
            variant
            and quality
        ):
            return None

        variant_highlight = (
            infer_highlight(
                data,
                variant,
                highlight,
            )
        )

        return {
            "category": variant,
            "value": quality,
            "highlight": variant_highlight,
            "sort": False,
            "title": "Ablation Study",
            "xlabel": quality,
        }

    # --------------------------------------------------------
    # Robustness heatmap
    # --------------------------------------------------------

    if figure_id == "robustness_heatmap":

        if not (
            model
            and scenario
            and quality
        ):
            return None

        # Dispatcher will pivot long-form data:
        #
        # Model | Scenario | mAP
        #
        # into:
        #
        # Model | Smoke | Blur | ...
        return {
            "row": model,
            "column": scenario,
            "value": quality,
            "title": "Robustness Heatmap",
        }

    # --------------------------------------------------------
    # Precision–Recall curve
    # --------------------------------------------------------

    if figure_id == "pr_curve":

        recall_axis = curve_column(
            inspection,
            "recall_axis",
        )

        precision_axis = curve_column(
            inspection,
            "precision_axis",
        )

        if not (
            recall_axis
            and precision_axis
        ):
            return None

        return {
            "x": recall_axis,
            "y": precision_axis,
            "group": model,
            "title": "Precision–Recall Curve",
            "xlabel": "Recall",
            "ylabel": "Precision",
        }

    # --------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------

    if figure_id == "roc_curve":

        fpr = curve_column(
            inspection,
            "false_positive_rate",
        )

        tpr = curve_column(
            inspection,
            "true_positive_rate",
        )

        if not (
            fpr
            and tpr
        ):
            return None

        return {
            "x": fpr,
            "y": tpr,
            "group": model,
            "title": "ROC Curve",
            "xlabel": "False Positive Rate",
            "ylabel": "True Positive Rate",
        }

    return None


# ------------------------------------------------------------
# Full figure planning
# ------------------------------------------------------------

def build_figure_plan(
    data: pd.DataFrame,
    inspection: dict[str, Any],
    selection: dict[str, Any],
    *,
    highlight: str | None = None,
    top: int = 3,
) -> dict[str, Any]:

    plans = []
    skipped = []

    recommendations = selection.get(
        "recommendations",
        [],
    )

    for recommendation in recommendations:

        figure_id = recommendation[
            "figure"
        ]
                # ----------------------------------------------------
        # Training recommendation can expand into
        # multiple scientifically compatible figures.
        # ----------------------------------------------------

        if figure_id == "training_curve":

            training_variants = (
                plan_training_variants(
                    data,
                    inspection,
                )
            )

            if not training_variants:

                skipped.append(
                    {
                        "figure": figure_id,
                        "reason": (
                            "No suitable training "
                            "curve variables could "
                            "be resolved."
                        ),
                    }
                )

                continue

            for variant in training_variants:

                plans.append(
                    {
                        "figure": variant[
                            "figure"
                        ],
                        "title": variant[
                            "title"
                        ],
                        "priority": recommendation[
                            "priority"
                        ],
                        "reason": recommendation[
                            "reason"
                        ],
                        "template": "line",
                        "parameters": variant[
                            "parameters"
                        ],
                    }
                )

                if len(plans) >= top:
                    break

            if len(plans) >= top:
                break

            continue

        template = (
            SEMANTIC_TO_TEMPLATE.get(
                figure_id
            )
        )

        # Scientific recommendation exists,
        # but our template library does not
        # currently know how to render it.
        if template is None:

            skipped.append(
                {
                    "figure": figure_id,
                    "reason": (
                        "No visual template is "
                        "registered yet."
                    ),
                }
            )

            continue

        parameters = plan_parameters(
            figure_id=figure_id,
            data=data,
            inspection=inspection,
            highlight=highlight,
        )

        if parameters is None:

            skipped.append(
                {
                    "figure": figure_id,
                    "template": template,
                    "reason": (
                        "Required plotting roles "
                        "could not be resolved."
                    ),
                }
            )

            continue

        plans.append(
            {
                "figure": figure_id,
                "title": recommendation[
                    "title"
                ],
                "priority": recommendation[
                    "priority"
                ],
                "reason": recommendation[
                    "reason"
                ],
                "template": template,
                "parameters": parameters,
            }
        )

        if len(plans) >= top:
            break

    return {
        "source": inspection.get(
            "source"
        ),
        "plans": plans,
        "skipped": skipped,
        "warnings": inspection.get(
            "warnings",
            [],
        ),
    }


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

def print_figure_plan(
    report: dict[str, Any],
) -> None:

    print(
        "sci-figure-maker figure planner"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Source: {report['source']}"
    )

    print()

    plans = report["plans"]

    if not plans:

        print(
            "No renderable figure plan "
            "could be generated."
        )

    else:

        print("Planned figures:")

        for index, plan in enumerate(
            plans,
            start=1,
        ):

            print()

            print(
                f"{index}. "
                f"{plan['title']}"
            )

            print(
                f"   Figure: "
                f"{plan['figure']}"
            )

            print(
                f"   Template: "
                f"{plan['template']}"
            )

            print(
                "   Parameters:"
            )

            for key, value in plan[
                "parameters"
            ].items():

                print(
                    f"     {key}: {value}"
                )

    if report["skipped"]:

        print()

        print("Skipped recommendations:")

        for item in report["skipped"]:

            print(
                f"  - {item['figure']}: "
                f"{item['reason']}"
            )


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
            "Convert scientific figure "
            "recommendations into concrete "
            "visual template plans."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input CSV, TSV, TXT, "
            "or XLSX file."
        ),
    )

    parser.add_argument(
        "--sheet",
        default="0",
        help=(
            "Excel sheet name or "
            "zero-based index."
        ),
    )

    parser.add_argument(
        "--highlight",
        default=None,
        help=(
            "Optional model or variant "
            "to emphasize, e.g. Ours."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help=(
            "Maximum number of "
            "renderable figure plans."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/planning/"
            "figure_plan.json"
        ),
        help=(
            "Output path for the "
            "figure plan JSON."
        ),
    )

    return parser


def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    data = load_data(
        input_path,
        sheet_name=parse_sheet(
            str(args.sheet)
        ),
    )

    inspection = inspect_data(
        data,
        source=input_path,
    )

    selection = build_selection_report(
        inspection
    )

    plan = build_figure_plan(
        data=data,
        inspection=inspection,
        selection=selection,
        highlight=args.highlight,
        top=args.top,
    )

    print_figure_plan(
        plan
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
        )

    print()

    print(
        "Figure plan saved to:"
    )

    print(
        f"  {output_path}"
    )


if __name__ == "__main__":
    main()