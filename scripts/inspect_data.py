from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.load_data import load_data
from scripts.theme import PROJECT_ROOT, load_yaml


PROFILE_FILE = (
    PROJECT_ROOT
    / "profiles"
    / "recognition_ai.yaml"
)


# ------------------------------------------------------------
# Name normalization
# ------------------------------------------------------------

def normalize_name(value: str) -> str:
    """
    Normalize column names and aliases for comparison.

    Examples
    --------
    "mAP50-95"   -> "map5095"
    "Params_M"   -> "paramsm"
    "Model Name" -> "modelname"
    """

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


# ------------------------------------------------------------
# Profile loading
# ------------------------------------------------------------

def load_recognition_profile() -> dict[str, Any]:
    """
    Load the Recognition AI semantic profile.
    """

    return load_yaml(PROFILE_FILE)


# ------------------------------------------------------------
# Column matching
# ------------------------------------------------------------

def alias_matches(
    column: str,
    canonical_name: str,
    aliases: list[str],
) -> bool:
    """
    Check whether a dataset column matches a semantic alias.
    """

    normalized_column = normalize_name(column)

    candidates = [
        canonical_name,
        *aliases,
    ]

    normalized_candidates = {
        normalize_name(value)
        for value in candidates
    }

    return normalized_column in normalized_candidates


def identify_semantic_roles(
    data: pd.DataFrame,
    profile: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Identify semantic roles such as:

    - model
    - epoch
    - class
    - scenario
    - variant
    """

    detected: dict[str, dict[str, Any]] = {}

    semantic_roles = profile.get(
        "semantic_roles",
        {},
    )

    for role_name, role_config in semantic_roles.items():

        aliases = role_config.get(
            "aliases",
            [],
        )

        for column in data.columns:

            if alias_matches(
                column,
                role_name,
                aliases,
            ):
                detected[role_name] = {
                    "column": column,
                }

                break

    return detected


def identify_metrics(
    data: pd.DataFrame,
    profile: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Identify known scientific metrics.

    Each detected metric records:

    - source column
    - metric family
    - optimization direction
    """

    detected: dict[str, dict[str, Any]] = {}

    metrics = profile.get(
        "metrics",
        {},
    )

    for metric_name, metric_config in metrics.items():

        aliases = metric_config.get(
            "aliases",
            [],
        )

        for column in data.columns:

            if alias_matches(
                column,
                metric_name,
                aliases,
            ):
                detected[metric_name] = {
                    "column": column,
                    "family": metric_config.get(
                        "family"
                    ),
                    "direction": metric_config.get(
                        "direction"
                    ),
                }

                break

    return detected


def identify_curve_variables(
    data: pd.DataFrame,
    profile: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Identify variables commonly used to form curves.
    """

    detected: dict[str, dict[str, Any]] = {}

    variables = profile.get(
        "curve_variables",
        {},
    )

    for variable_name, variable_config in variables.items():

        aliases = variable_config.get(
            "aliases",
            [],
        )

        for column in data.columns:

            if alias_matches(
                column,
                variable_name,
                aliases,
            ):
                detected[variable_name] = {
                    "column": column,
                }

                break

    return detected


# ------------------------------------------------------------
# Basic data structure
# ------------------------------------------------------------

def classify_columns(
    data: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Classify columns into broad structural types.
    """

    numeric = []
    categorical = []
    datetime = []
    other = []

    for column in data.columns:

        series = data[column]

        if pd.api.types.is_numeric_dtype(series):
            numeric.append(column)

        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime.append(column)

        elif (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            categorical.append(column)

        else:
            other.append(column)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime,
        "other": other,
    }


# ------------------------------------------------------------
# Experiment inference
# ------------------------------------------------------------

def infer_experiment_types(
    data: pd.DataFrame,
    roles: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    curves: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Infer likely experiment types from data structure.

    These are recommendations, not scientific conclusions.
    """

    candidates: list[dict[str, Any]] = []

    has_model = "model" in roles
    has_epoch = "epoch" in roles
    has_variant = "variant" in roles
    has_scenario = "scenario" in roles

    metric_families = {
        info.get("family")
        for info in metrics.values()
    }

    has_quality = "quality" in metric_families
    has_efficiency = "efficiency" in metric_families
    has_complexity = "complexity" in metric_families
    has_training = "training" in metric_families

    # Training experiment
    if has_epoch and has_training:

        candidates.append(
            {
                "type": "training",
                "confidence": "high",
                "reason": (
                    "Epoch-like variable and training "
                    "metric were detected."
                ),
            }
        )

    # Model benchmark
    if has_model and has_quality:

        candidates.append(
            {
                "type": "benchmark",
                "confidence": "high",
                "reason": (
                    "Model identity and one or more "
                    "quality metrics were detected."
                ),
            }
        )

    # Accuracy-efficiency / complexity comparison
    if (
        has_model
        and has_quality
        and (
            has_efficiency
            or has_complexity
        )
    ):

        candidates.append(
            {
                "type": "accuracy_efficiency",
                "confidence": "high",
                "reason": (
                    "Model, quality, and efficiency/"
                    "complexity metrics were detected."
                ),
            }
        )

    # Ablation study
    if has_variant and has_quality:

        candidates.append(
            {
                "type": "ablation",
                "confidence": "high",
                "reason": (
                    "Variant/configuration column and "
                    "quality metric were detected."
                ),
            }
        )

    # Robustness experiment
    if (
        has_model
        and has_scenario
        and has_quality
    ):

        candidates.append(
            {
                "type": "robustness",
                "confidence": "high",
                "reason": (
                    "Model, scenario, and quality "
                    "metric were detected."
                ),
            }
        )

    # PR curve:
    # Precision + Recall alone are NOT enough.
    # A benchmark table may contain only one P/R pair per model.
    if (
        "precision_axis" in curves
        and "recall_axis" in curves
    ):

        repeated_model_measurements = False

        if has_model:

            model_column = roles["model"]["column"]

            repeated_model_measurements = (
                data[model_column]
                .duplicated()
                .any()
            )

        if (
            repeated_model_measurements
            or (
                not has_model
                and len(data) >= 5
            )
        ):

            candidates.append(
                {
                    "type": "pr_curve",
                    "confidence": "medium",
                    "reason": (
                        "Precision and Recall series "
                        "appear to contain multiple "
                        "curve points."
                    ),
                }
            )

    # ROC curve
    if (
        "false_positive_rate" in curves
        and "true_positive_rate" in curves
    ):

        candidates.append(
            {
                "type": "roc_curve",
                "confidence": "medium",
                "reason": (
                    "FPR and TPR variables were detected."
                ),
            }
        )

    return candidates


# ------------------------------------------------------------
# Warnings
# ------------------------------------------------------------

def build_warnings(
    data: pd.DataFrame,
    metrics: dict[str, dict[str, Any]],
) -> list[str]:
    """
    Generate simple data-quality warnings.
    """

    warnings: list[str] = []

    missing_total = int(
        data.isna().sum().sum()
    )

    if missing_total > 0:
        warnings.append(
            f"Dataset contains {missing_total} missing values."
        )

    duplicate_rows = int(
        data.duplicated().sum()
    )

    if duplicate_rows > 0:
        warnings.append(
            f"Dataset contains {duplicate_rows} duplicated rows."
        )

    if not metrics:
        warnings.append(
            "No known Recognition AI metrics were detected."
        )

    return warnings


# ------------------------------------------------------------
# Full inspection
# ------------------------------------------------------------

def inspect_data(
    data: pd.DataFrame,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """
    Inspect a scientific dataset and return a structured report.
    """

    profile = load_recognition_profile()

    roles = identify_semantic_roles(
        data,
        profile,
    )

    metrics = identify_metrics(
        data,
        profile,
    )

    curves = identify_curve_variables(
        data,
        profile,
    )

    column_types = classify_columns(
        data
    )

    experiment_types = infer_experiment_types(
        data=data,
        roles=roles,
        metrics=metrics,
        curves=curves,
    )

    warnings = build_warnings(
        data,
        metrics,
    )

    report = {
        "source": (
            str(source)
            if source is not None
            else None
        ),
        "shape": {
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
        },
        "columns": list(data.columns),
        "column_types": column_types,
        "missing_values": {
            column: int(data[column].isna().sum())
            for column in data.columns
        },
        "duplicate_rows": int(
            data.duplicated().sum()
        ),
        "semantic_roles": roles,
        "metrics": metrics,
        "curve_variables": curves,
        "candidate_experiments": experiment_types,
        "warnings": warnings,
    }

    return report


# ------------------------------------------------------------
# Human-readable output
# ------------------------------------------------------------

def print_report(
    report: dict[str, Any],
) -> None:
    """
    Print a concise human-readable inspection report.
    """

    print("sci-figure-maker data inspector")
    print("-------------------------------")

    print(
        f"Source: {report['source']}"
    )

    print(
        "Shape:",
        f"{report['shape']['rows']} rows x",
        f"{report['shape']['columns']} columns",
    )

    print()

    print("Semantic roles:")

    if report["semantic_roles"]:

        for role, info in report[
            "semantic_roles"
        ].items():

            print(
                f"  - {role}: "
                f"{info['column']}"
            )

    else:
        print("  None detected")

    print()

    print("Metrics:")

    if report["metrics"]:

        for metric, info in report[
            "metrics"
        ].items():

            print(
                f"  - {metric}: "
                f"{info['column']} "
                f"[{info['family']}, "
                f"{info['direction']}]"
            )

    else:
        print("  None detected")

    print()

    print("Candidate experiments:")

    if report["candidate_experiments"]:

        for candidate in report[
            "candidate_experiments"
        ]:

            print(
                f"  - {candidate['type']} "
                f"({candidate['confidence']})"
            )

            print(
                f"    {candidate['reason']}"
            )

    else:
        print("  None detected")

    print()

    print(
        f"Duplicate rows: "
        f"{report['duplicate_rows']}"
    )

    total_missing = sum(
        report["missing_values"].values()
    )

    print(
        f"Missing values: {total_missing}"
    )

    if report["warnings"]:

        print()
        print("Warnings:")

        for warning in report["warnings"]:
            print(f"  - {warning}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Inspect scientific experimental data "
            "and infer Recognition AI semantics."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV, TSV, TXT, or XLSX file.",
    )

    parser.add_argument(
        "--sheet",
        default=0,
        help=(
            "Excel sheet name or zero-based index."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/inspection/"
            "data_report.json"
        ),
        help=(
            "Path for the generated JSON report."
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

    parser = build_parser()

    args = parser.parse_args()

    input_path = Path(args.input)

    data = load_data(
        input_path,
        sheet_name=parse_sheet_argument(
            str(args.sheet)
        ),
    )

    report = inspect_data(
        data,
        source=input_path,
    )

    print_report(report)

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"JSON report saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()