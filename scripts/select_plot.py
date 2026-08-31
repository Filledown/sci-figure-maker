from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.inspect_data import inspect_data
from scripts.load_data import load_data


# ------------------------------------------------------------
# Figure catalog
# ------------------------------------------------------------

FIGURE_CATALOG: dict[str, dict[str, Any]] = {

    "model_comparison": {
        "title": "Model Comparison",
        "description": (
            "Compare one or more performance metrics "
            "across multiple models."
        ),
        "implemented": True,
    },

    "model_ranking": {
        "title": "Model Ranking",
        "description": (
            "Rank models according to a selected metric."
        ),
        "implemented": True,
    },

    "accuracy_efficiency": {
        "title": "Accuracy–Efficiency Trade-off",
        "description": (
            "Compare predictive performance against "
            "speed or computational efficiency."
        ),
        "implemented": True,
    },

    "pareto": {
        "title": "Pareto Frontier",
        "description": (
            "Identify non-dominated models across "
            "quality and efficiency/complexity metrics."
        ),
        "implemented": False,
    },

    "training_curve": {
        "title": "Training Curve",
        "description": (
            "Visualize training or validation metrics "
            "across epochs or iterations."
        ),
        "implemented": True,
    },

    "confidence_curve": {
        "title": "Confidence Curves",
        "description": (
            "Visualize Precision, Recall, and F1 "
            "as functions of the confidence threshold."
        ),
        "implemented": True,
    },

    "ablation": {
        "title": "Ablation Study",
        "description": (
            "Show the performance changes associated "
            "with model components or configurations."
        ),
        "implemented": True,
    },

    "robustness_heatmap": {
        "title": "Robustness Heatmap",
        "description": (
            "Compare model performance under multiple "
            "experimental conditions or scenarios."
        ),
        "implemented": True,
    },

    "pr_curve": {
        "title": "Precision–Recall Curve",
        "description": (
            "Visualize the Precision–Recall trade-off "
            "using real curve-point data."
        ),
        "implemented": True,
    },

    "roc_curve": {
        "title": "ROC Curve",
        "description": (
            "Visualize true-positive rate against "
            "false-positive rate."
        ),
        "implemented": True,
    },
}


# ------------------------------------------------------------
# Recommendation rules
# ------------------------------------------------------------

EXPERIMENT_FIGURE_RULES = {

    "benchmark": [
        {
            "figure": "model_comparison",
            "priority": 90,
            "reason": (
                "The dataset contains model identities "
                "and model-quality metrics."
            ),
        },
        {
            "figure": "model_ranking",
            "priority": 70,
            "reason": (
                "Ranking can clarify relative performance "
                "when one metric is especially important."
            ),
        },
    ],

    "accuracy_efficiency": [
        {
            "figure": "accuracy_efficiency",
            "priority": 100,
            "reason": (
                "The dataset contains model-quality and "
                "efficiency or complexity metrics."
            ),
        },
        {
            "figure": "pareto",
            "priority": 85,
            "reason": (
                "Pareto analysis can reveal models that "
                "offer non-dominated quality-efficiency "
                "trade-offs."
            ),
        },
    ],

    "training": [
        {
            "figure": "training_curve",
            "priority": 100,
            "reason": (
                "Epoch-like and training-related "
                "variables were detected."
            ),
        },
    ],

    "confidence_curve": [
        {
            "figure": "confidence_curve",
            "priority": 100,
            "reason": (
                "Confidence threshold and Precision, "
                "Recall, or F1 series were detected."
            ),
        },
    ],

    "ablation": [
        {
            "figure": "ablation",
            "priority": 100,
            "reason": (
                "The dataset appears to compare "
                "model variants or configurations."
            ),
        },
    ],

    "robustness": [
        {
            "figure": "robustness_heatmap",
            "priority": 100,
            "reason": (
                "The dataset contains models, scenarios, "
                "and quality metrics."
            ),
        },
    ],

    "pr_curve": [
        {
            "figure": "pr_curve",
            "priority": 100,
            "reason": (
                "Multiple Precision–Recall curve points "
                "appear to be available."
            ),
        },
    ],

    "roc_curve": [
        {
            "figure": "roc_curve",
            "priority": 100,
            "reason": (
                "False-positive-rate and true-positive-rate "
                "variables were detected."
            ),
        },
    ],
}


# ------------------------------------------------------------
# Recommendation engine
# ------------------------------------------------------------

def recommend_figures(
    inspection_report: dict[str, Any],
) -> list[dict[str, Any]]:

    recommendations: dict[
        str,
        dict[str, Any],
    ] = {}

    candidate_experiments = (
        inspection_report.get(
            "candidate_experiments",
            [],
        )
    )

    for experiment in candidate_experiments:

        experiment_type = experiment[
            "type"
        ]

        rules = EXPERIMENT_FIGURE_RULES.get(
            experiment_type,
            [],
        )

        for rule in rules:

            figure_name = rule[
                "figure"
            ]

            catalog_entry = (
                FIGURE_CATALOG[
                    figure_name
                ]
            )

            recommendation = {
                "figure": figure_name,
                "title": catalog_entry[
                    "title"
                ],
                "priority": rule[
                    "priority"
                ],
                "implemented": (
                    catalog_entry[
                        "implemented"
                    ]
                ),
                "experiment_type": (
                    experiment_type
                ),
                "reason": rule[
                    "reason"
                ],
                "description": (
                    catalog_entry[
                        "description"
                    ]
                ),
            }

            if (
                figure_name
                not in recommendations
            ):

                recommendations[
                    figure_name
                ] = recommendation

            else:

                existing = (
                    recommendations[
                        figure_name
                    ]
                )

                if (
                    recommendation[
                        "priority"
                    ]
                    > existing[
                        "priority"
                    ]
                ):

                    recommendations[
                        figure_name
                    ] = recommendation

    return sorted(
        recommendations.values(),
        key=lambda item: item[
            "priority"
        ],
        reverse=True,
    )


def choose_primary_figure(
    recommendations: list[
        dict[str, Any]
    ],
) -> dict[str, Any] | None:

    if not recommendations:
        return None

    return recommendations[0]


def build_selection_report(
    inspection_report: dict[str, Any],
) -> dict[str, Any]:

    recommendations = (
        recommend_figures(
            inspection_report
        )
    )

    primary = choose_primary_figure(
        recommendations
    )

    return {
        "source": inspection_report.get(
            "source"
        ),
        "candidate_experiments": (
            inspection_report.get(
                "candidate_experiments",
                [],
            )
        ),
        "primary_recommendation": (
            primary
        ),
        "recommendations": (
            recommendations
        ),
        "warnings": (
            inspection_report.get(
                "warnings",
                [],
            )
        ),
    }


# ------------------------------------------------------------
# Human-readable output
# ------------------------------------------------------------

def print_selection_report(
    report: dict[str, Any],
) -> None:

    print(
        "sci-figure-maker plot selector"
    )

    print(
        "------------------------------"
    )

    print(
        f"Source: {report['source']}"
    )

    print()

    primary = report[
        "primary_recommendation"
    ]

    if primary is None:

        print(
            "No confident figure recommendation "
            "could be generated."
        )

        return

    print(
        "Primary recommendation:"
    )

    print(
        f"  {primary['title']}"
    )

    print(
        f"  Figure ID: "
        f"{primary['figure']}"
    )

    print(
        f"  Implemented: "
        f"{primary['implemented']}"
    )

    print(
        f"  Reason: "
        f"{primary['reason']}"
    )

    print()

    print(
        "All recommendations:"
    )

    for index, item in enumerate(
        report["recommendations"],
        start=1,
    ):

        status = (
            "READY"
            if item["implemented"]
            else "PLANNED"
        )

        print(
            f"  {index}. "
            f"{item['title']} "
            f"[{status}]"
        )

        print(
            f"     Priority: "
            f"{item['priority']}"
        )

        print(
            f"     {item['reason']}"
        )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Automatically recommend scientific "
            "figure types from experimental data."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--sheet",
        default=0,
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/selection/"
            "figure_recommendations.json"
        ),
    )

    return parser


def parse_sheet_argument(
    value: str,
) -> str | int:

    try:
        return int(value)

    except ValueError:
        return value


def main() -> None:

    args = build_parser().parse_args()

    input_path = Path(
        args.input
    )

    data = load_data(
        input_path,
        sheet_name=parse_sheet_argument(
            str(args.sheet)
        ),
    )

    inspection_report = inspect_data(
        data,
        source=input_path,
    )

    selection_report = (
        build_selection_report(
            inspection_report
        )
    )

    print_selection_report(
        selection_report
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
            selection_report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()

    print(
        "Recommendation report saved to:"
    )

    print(
        f"  {output_path}"
    )


if __name__ == "__main__":
    main()