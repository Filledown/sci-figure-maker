from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Generic numerical helpers
# ------------------------------------------------------------

def clean_xy(
    data: pd.DataFrame,
    x: str,
    y: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract finite numeric x/y values.

    Missing or non-numeric values are removed.
    Original observations are never modified.
    """

    working = data[
        [
            x,
            y,
        ]
    ].copy()

    working[x] = pd.to_numeric(
        working[x],
        errors="coerce",
    )

    working[y] = pd.to_numeric(
        working[y],
        errors="coerce",
    )

    working = working.dropna(
        subset=[
            x,
            y,
        ]
    )

    x_values = working[x].to_numpy(
        dtype=float
    )

    y_values = working[y].to_numpy(
        dtype=float
    )

    finite = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
    )

    return (
        x_values[finite],
        y_values[finite],
    )


def trapezoidal_area(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> float:
    """
    Calculate area under a curve using trapezoidal integration.

    X values are sorted for numerical integration.
    No synthetic points are inserted.
    """

    if len(x_values) < 2:
        raise ValueError(
            "At least two curve points are required."
        )

    order = np.argsort(
        x_values
    )

    x_sorted = x_values[
        order
    ]

    y_sorted = y_values[
        order
    ]

    widths = (
        x_sorted[1:]
        - x_sorted[:-1]
    )

    heights = (
        y_sorted[1:]
        + y_sorted[:-1]
    ) / 2.0

    return float(
        np.sum(
            widths * heights
        )
    )


def validate_unit_interval(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_name: str,
    y_name: str,
    series_name: str,
) -> None:
    """
    Validate metrics that must lie in [0, 1].
    """

    tolerance = 1e-9

    if len(x_values) == 0:
        raise ValueError(
            f"Curve '{series_name}' contains "
            "no valid numeric points."
        )

    if (
        np.min(x_values) < -tolerance
        or np.max(x_values) > 1.0 + tolerance
    ):
        raise ValueError(
            f"Curve '{series_name}' contains "
            f"{x_name} values outside [0, 1]."
        )

    if (
        np.min(y_values) < -tolerance
        or np.max(y_values) > 1.0 + tolerance
    ):
        raise ValueError(
            f"Curve '{series_name}' contains "
            f"{y_name} values outside [0, 1]."
        )


# ------------------------------------------------------------
# ROC semantics
# ------------------------------------------------------------

def validate_roc(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    series_name: str,
) -> None:

    validate_unit_interval(
        x_values,
        y_values,
        x_name="FPR",
        y_name="TPR",
        series_name=series_name,
    )


def warn_roc_endpoints(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    series_name: str,
) -> None:
    """
    Warn about absent standard ROC endpoints.

    Endpoints are NEVER added automatically.
    """

    has_origin = bool(
        np.any(
            np.isclose(
                x_values,
                0.0,
            )
            & np.isclose(
                y_values,
                0.0,
            )
        )
    )

    has_end = bool(
        np.any(
            np.isclose(
                x_values,
                1.0,
            )
            & np.isclose(
                y_values,
                1.0,
            )
        )
    )

    if not has_origin:
        warnings.warn(
            (
                f"ROC series '{series_name}' does not "
                "contain (0, 0). No point was added."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    if not has_end:
        warnings.warn(
            (
                f"ROC series '{series_name}' does not "
                "contain (1, 1). No point was added."
            ),
            RuntimeWarning,
            stacklevel=2,
        )


def roc_label(
    series_name: str,
    auc_value: float,
) -> str:

    return (
        f"{series_name} "
        f"(AUC = {auc_value:.3f})"
    )


# ------------------------------------------------------------
# Precision-Recall semantics
# ------------------------------------------------------------

def validate_pr(
    recall_values: np.ndarray,
    precision_values: np.ndarray,
    *,
    series_name: str,
) -> None:

    validate_unit_interval(
        recall_values,
        precision_values,
        x_name="Recall",
        y_name="Precision",
        series_name=series_name,
    )


def pr_area(
    recall_values: np.ndarray,
    precision_values: np.ndarray,
) -> float:
    """
    Trapezoidal area under the Precision-Recall curve.

    Important:
    This is NOT automatically equivalent to the AP reported
    by a particular object-detection/classification framework.

    AP may use framework-specific interpolation/integration
    conventions. Therefore this value is explicitly treated
    as trapezoidal AUPRC.
    """

    return trapezoidal_area(
        recall_values,
        precision_values,
    )


def pr_label(
    series_name: str,
    auprc_value: float,
) -> str:
    """
    Explicitly label the numerical method to avoid pretending
    that trapezoidal AUPRC is necessarily framework-reported AP.
    """

    return (
        f"{series_name} "
        f"(AUPRC = {auprc_value:.3f})"
    )


def prepare_grouped_curve(
    data: pd.DataFrame,
    *,
    group: str,
    x: str,
    y: str,
    curve_type: str,
) -> tuple[
    pd.DataFrame,
    list[Any],
    list[str],
]:
    """
    Prepare long-form multi-model curve data.

    Returns:
        wide dataframe
        curve columns
        publication legend labels
    """

    working = data[
        [
            group,
            x,
            y,
        ]
    ].copy()

    working[x] = pd.to_numeric(
        working[x],
        errors="coerce",
    )

    working[y] = pd.to_numeric(
        working[y],
        errors="coerce",
    )

    working = working.dropna(
        subset=[
            group,
            x,
            y,
        ]
    )

    # Only row ordering is changed.
    # Measurement values remain untouched.
    working = working.sort_values(
        by=[
            group,
            x,
        ],
        kind="stable",
    )

    label_lookup: dict[str, str] = {}

    for group_value, subset in working.groupby(
        group,
        sort=False,
    ):

        series_name = str(
            group_value
        )

        x_values, y_values = clean_xy(
            subset,
            x,
            y,
        )

        if curve_type == "roc":

            validate_roc(
                x_values,
                y_values,
                series_name=series_name,
            )

            warn_roc_endpoints(
                x_values,
                y_values,
                series_name=series_name,
            )

            area = trapezoidal_area(
                x_values,
                y_values,
            )

            label_lookup[
                series_name
            ] = roc_label(
                series_name,
                area,
            )

        elif curve_type == "pr":

            validate_pr(
                x_values,
                y_values,
                series_name=series_name,
            )

            area = pr_area(
                x_values,
                y_values,
            )

            label_lookup[
                series_name
            ] = pr_label(
                series_name,
                area,
            )

        else:

            raise ValueError(
                f"Unsupported curve type: "
                f"{curve_type}"
            )

    wide = (
        working
        .pivot_table(
            index=x,
            columns=group,
            values=y,
            aggfunc="mean",
        )
        .reset_index()
    )

    curve_columns = [
        column
        for column in wide.columns
        if column != x
    ]

    labels = [
        label_lookup.get(
            str(column),
            str(column),
        )
        for column in curve_columns
    ]

    return (
        wide,
        curve_columns,
        labels,
    )