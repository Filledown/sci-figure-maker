from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.figure_planner import build_figure_plan
from scripts.inspect_data import inspect_data
from scripts.load_data import load_data
from scripts.select_plot import build_selection_report
from scripts.template_registry import TEMPLATE_REGISTRY


# ------------------------------------------------------------
# Numerical helpers
# ------------------------------------------------------------

def _clean_xy(
    data: pd.DataFrame,
    x: str,
    y: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return finite numeric X/Y values.

    The function removes only missing/non-finite points.
    It never invents or modifies observations.
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


def _trapezoidal_auc(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> float:
    """
    Calculate area under a curve using the trapezoidal rule.

    X values are sorted before integration.
    Data values themselves are never modified.
    """

    if len(x_values) < 2:

        raise ValueError(
            "At least two curve points are "
            "required to calculate AUC."
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
    ) * 0.5

    return float(
        np.sum(
            widths * heights
        )
    )


def _validate_roc_range(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    series_name: str,
) -> None:
    """
    Validate that FPR and TPR lie inside [0, 1].
    """

    tolerance = 1e-9

    if len(x_values) == 0:

        raise ValueError(
            f"ROC series '{series_name}' "
            f"contains no valid points."
        )

    if (
        np.min(x_values)
        < -tolerance
        or np.max(x_values)
        > 1.0 + tolerance
    ):

        raise ValueError(
            f"ROC series '{series_name}' "
            f"contains FPR values outside [0, 1]."
        )

    if (
        np.min(y_values)
        < -tolerance
        or np.max(y_values)
        > 1.0 + tolerance
    ):

        raise ValueError(
            f"ROC series '{series_name}' "
            f"contains TPR values outside [0, 1]."
        )


def _validate_pr_range(
    recall_values: np.ndarray,
    precision_values: np.ndarray,
    *,
    series_name: str,
) -> None:
    """
    Validate that Recall and Precision lie inside [0, 1].
    """

    tolerance = 1e-9

    if len(recall_values) == 0:

        raise ValueError(
            f"PR series '{series_name}' "
            f"contains no valid points."
        )

    if (
        np.min(recall_values)
        < -tolerance
        or np.max(recall_values)
        > 1.0 + tolerance
    ):

        raise ValueError(
            f"PR series '{series_name}' "
            f"contains Recall values outside [0, 1]."
        )

    if (
        np.min(precision_values)
        < -tolerance
        or np.max(precision_values)
        > 1.0 + tolerance
    ):

        raise ValueError(
            f"PR series '{series_name}' "
            f"contains Precision values outside [0, 1]."
        )


def _warn_roc_endpoints(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    series_name: str,
) -> None:
    """
    Warn when standard ROC endpoints are not present.

    Important:
    The function NEVER inserts (0, 0) or (1, 1).
    """

    if len(x_values) == 0:

        return

    start_present = bool(
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

    end_present = bool(
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

    if not start_present:

        warnings.warn(
            (
                f"ROC series '{series_name}' "
                "does not contain the (0, 0) "
                "endpoint. The endpoint was NOT "
                "added automatically."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    if not end_present:

        warnings.warn(
            (
                f"ROC series '{series_name}' "
                "does not contain the (1, 1) "
                "endpoint. The endpoint was NOT "
                "added automatically."
            ),
            RuntimeWarning,
            stacklevel=2,
        )


def _roc_label(
    name: str,
    auc_value: float,
) -> str:
    """
    Format a publication-style ROC legend label.
    """

    return (
        f"{name} "
        f"(AUC = {auc_value:.3f})"
    )


def _pr_label(
    name: str,
    auprc_value: float,
) -> str:
    """
    Format a publication-style PR legend label.

    We intentionally use AUPRC instead of AP because
    trapezoidal integration is not necessarily identical
    to a framework-reported Average Precision metric.
    """

    return (
        f"{name} "
        f"(AUPRC = {auprc_value:.3f})"
    )


# ------------------------------------------------------------
# Line-data preparation
# ------------------------------------------------------------

def prepare_line_data(
    data: pd.DataFrame,
    parameters: dict[str, Any],
    *,
    figure_id: str | None = None,
) -> dict[str, Any]:
    """
    Convert curve plans into arguments accepted by
    templates.line.render_line().

    ROC figures receive additional scientific handling:

    - FPR/TPR validation
    - automatic AUC calculation
    - chance diagonal
    - fixed [0, 1] axes
    - endpoint warnings

    PR figures receive additional scientific handling:

    - Recall/Precision validation
    - automatic trapezoidal AUPRC calculation
    - fixed [0, 1] axes
    - no artificial baseline
    """

    x = parameters.get(
        "x"
    )

    y = parameters.get(
        "y"
    )

    ys = parameters.get(
        "ys"
    )

    group = parameters.get(
        "group"
    )

    is_roc = (
        figure_id
        == "roc_curve"
    )

    is_pr = (
        figure_id
        == "pr_curve"
    )

    # --------------------------------------------------------
    # Wide-form multi-line data
    # --------------------------------------------------------

    if ys:

        labels = parameters.get(
            "labels"
        )

        if labels is None:

            labels = [
                str(column)
                for column
                in ys
            ]

        else:

            labels = list(
                labels
            )

        result = {
            "data": data,
            "x": x,
            "ys": ys,
            "labels": labels,
            "highlight": parameters.get(
                "highlight"
            ),
            "title": parameters.get(
                "title"
            ),
            "xlabel": parameters.get(
                "xlabel"
            ),
            "ylabel": parameters.get(
                "ylabel"
            ),
        }

        if is_roc:

            roc_labels = []

            for column, label in zip(
                ys,
                labels,
            ):

                x_values, y_values = (
                    _clean_xy(
                        data,
                        x,
                        column,
                    )
                )

                _validate_roc_range(
                    x_values,
                    y_values,
                    series_name=str(
                        label
                    ),
                )

                _warn_roc_endpoints(
                    x_values,
                    y_values,
                    series_name=str(
                        label
                    ),
                )

                auc_value = (
                    _trapezoidal_auc(
                        x_values,
                        y_values,
                    )
                )

                roc_labels.append(
                    _roc_label(
                        str(label),
                        auc_value,
                    )
                )

            result[
                "labels"
            ] = roc_labels

            result[
                "xlabel"
            ] = (
                parameters.get(
                    "xlabel"
                )
                or "False Positive Rate"
            )

            result[
                "ylabel"
            ] = (
                parameters.get(
                    "ylabel"
                )
                or "True Positive Rate"
            )

            result[
                "xlim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "ylim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "reference_diagonal"
            ] = True

            result[
                "sort_x"
            ] = True

        if is_pr:

            pr_labels = []

            for column, label in zip(
                ys,
                labels,
            ):

                recall_values, precision_values = (
                    _clean_xy(
                        data,
                        x,
                        column,
                    )
                )

                _validate_pr_range(
                    recall_values,
                    precision_values,
                    series_name=str(
                        label
                    ),
                )

                auprc_value = (
                    _trapezoidal_auc(
                        recall_values,
                        precision_values,
                    )
                )

                pr_labels.append(
                    _pr_label(
                        str(label),
                        auprc_value,
                    )
                )

            result[
                "labels"
            ] = pr_labels

            result[
                "xlabel"
            ] = (
                parameters.get(
                    "xlabel"
                )
                or "Recall"
            )

            result[
                "ylabel"
            ] = (
                parameters.get(
                    "ylabel"
                )
                or "Precision"
            )

            result[
                "xlim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "ylim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "reference_diagonal"
            ] = False

            result[
                "sort_x"
            ] = True

        return result

    # --------------------------------------------------------
    # Long-form grouped curve
    #
    # Example ROC:
    # Model | FPR | TPR
    #
    # Example PR:
    # Model | Recall | Precision
    # --------------------------------------------------------

    if (
        group
        and x
        and y
    ):

        working = data[
            [
                group,
                x,
                y,
            ]
        ].copy()

        working = working.dropna(
            subset=[
                group,
                x,
                y,
            ]
        )

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

        if is_roc or is_pr:

            working = (
                working
                .sort_values(
                    by=[
                        group,
                        x,
                    ],
                    kind="stable",
                )
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
            str(column)
            for column
            in curve_columns
        ]

        result = {
            "data": wide,
            "x": x,
            "ys": curve_columns,
            "labels": labels,
            "highlight": parameters.get(
                "highlight"
            ),
            "title": parameters.get(
                "title"
            ),
            "xlabel": parameters.get(
                "xlabel"
            ),
            "ylabel": parameters.get(
                "ylabel"
            ),
        }

        if is_roc:

            auc_lookup: dict[
                str,
                float,
            ] = {}

            for group_value, subset in (
                working.groupby(
                    group,
                    sort=False,
                )
            ):

                x_values, y_values = (
                    _clean_xy(
                        subset,
                        x,
                        y,
                    )
                )

                series_name = str(
                    group_value
                )

                _validate_roc_range(
                    x_values,
                    y_values,
                    series_name=series_name,
                )

                _warn_roc_endpoints(
                    x_values,
                    y_values,
                    series_name=series_name,
                )

                auc_lookup[
                    series_name
                ] = (
                    _trapezoidal_auc(
                        x_values,
                        y_values,
                    )
                )

            roc_labels = []

            for column in curve_columns:

                series_name = str(
                    column
                )

                auc_value = (
                    auc_lookup.get(
                        series_name
                    )
                )

                if auc_value is None:

                    roc_labels.append(
                        series_name
                    )

                else:

                    roc_labels.append(
                        _roc_label(
                            series_name,
                            auc_value,
                        )
                    )

            result[
                "labels"
            ] = roc_labels

            result[
                "xlabel"
            ] = (
                parameters.get(
                    "xlabel"
                )
                or "False Positive Rate"
            )

            result[
                "ylabel"
            ] = (
                parameters.get(
                    "ylabel"
                )
                or "True Positive Rate"
            )

            result[
                "xlim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "ylim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "reference_diagonal"
            ] = True

            result[
                "sort_x"
            ] = True

        if is_pr:

            auprc_lookup: dict[
                str,
                float,
            ] = {}

            for group_value, subset in (
                working.groupby(
                    group,
                    sort=False,
                )
            ):

                recall_values, precision_values = (
                    _clean_xy(
                        subset,
                        x,
                        y,
                    )
                )

                series_name = str(
                    group_value
                )

                _validate_pr_range(
                    recall_values,
                    precision_values,
                    series_name=series_name,
                )

                auprc_lookup[
                    series_name
                ] = (
                    _trapezoidal_auc(
                        recall_values,
                        precision_values,
                    )
                )

            pr_labels = []

            for column in curve_columns:

                series_name = str(
                    column
                )

                auprc_value = (
                    auprc_lookup.get(
                        series_name
                    )
                )

                if auprc_value is None:

                    pr_labels.append(
                        series_name
                    )

                else:

                    pr_labels.append(
                        _pr_label(
                            series_name,
                            auprc_value,
                        )
                    )

            result[
                "labels"
            ] = pr_labels

            result[
                "xlabel"
            ] = (
                parameters.get(
                    "xlabel"
                )
                or "Recall"
            )

            result[
                "ylabel"
            ] = (
                parameters.get(
                    "ylabel"
                )
                or "Precision"
            )

            result[
                "xlim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "ylim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "reference_diagonal"
            ] = False

            result[
                "sort_x"
            ] = True

        return result

    # --------------------------------------------------------
    # Single curve
    # --------------------------------------------------------

    if (
        x
        and y
    ):

        labels = parameters.get(
            "labels"
        )

        if labels is None:

            labels = [
                str(y)
            ]

        else:

            labels = list(
                labels
            )

        result = {
            "data": data,
            "x": x,
            "ys": [
                y
            ],
            "labels": labels,
            "highlight": parameters.get(
                "highlight"
            ),
            "title": parameters.get(
                "title"
            ),
            "xlabel": parameters.get(
                "xlabel"
            ),
            "ylabel": parameters.get(
                "ylabel"
            ),
        }

        if is_roc:

            x_values, y_values = (
                _clean_xy(
                    data,
                    x,
                    y,
                )
            )

            series_name = (
                str(labels[0])
                if labels
                else str(y)
            )

            _validate_roc_range(
                x_values,
                y_values,
                series_name=series_name,
            )

            _warn_roc_endpoints(
                x_values,
                y_values,
                series_name=series_name,
            )

            auc_value = (
                _trapezoidal_auc(
                    x_values,
                    y_values,
                )
            )

            result[
                "labels"
            ] = [
                _roc_label(
                    series_name,
                    auc_value,
                )
            ]

            result[
                "xlabel"
            ] = (
                parameters.get(
                    "xlabel"
                )
                or "False Positive Rate"
            )

            result[
                "ylabel"
            ] = (
                parameters.get(
                    "ylabel"
                )
                or "True Positive Rate"
            )

            result[
                "xlim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "ylim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "reference_diagonal"
            ] = True

            result[
                "sort_x"
            ] = True

        if is_pr:

            recall_values, precision_values = (
                _clean_xy(
                    data,
                    x,
                    y,
                )
            )

            series_name = (
                str(labels[0])
                if labels
                else str(y)
            )

            _validate_pr_range(
                recall_values,
                precision_values,
                series_name=series_name,
            )

            auprc_value = (
                _trapezoidal_auc(
                    recall_values,
                    precision_values,
                )
            )

            result[
                "labels"
            ] = [
                _pr_label(
                    series_name,
                    auprc_value,
                )
            ]

            result[
                "xlabel"
            ] = (
                parameters.get(
                    "xlabel"
                )
                or "Recall"
            )

            result[
                "ylabel"
            ] = (
                parameters.get(
                    "ylabel"
                )
                or "Precision"
            )

            result[
                "xlim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "ylim"
            ] = (
                0.0,
                1.0,
            )

            result[
                "reference_diagonal"
            ] = False

            result[
                "sort_x"
            ] = True

        return result

    raise ValueError(
        "Unable to prepare line-plot data."
    )


# ------------------------------------------------------------
# Heatmap-data preparation
# ------------------------------------------------------------

def prepare_heatmap_data(
    data: pd.DataFrame,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert long-form robustness data:

    Model | Scenario | Metric

    into wide-form:

    Model | Smoke | Blur | Occlusion | ...
    """

    row = parameters.get(
        "row"
    )

    column = parameters.get(
        "column"
    )

    value = parameters.get(
        "value"
    )

    if not (
        row
        and column
        and value
    ):

        raise ValueError(
            "Heatmap plan requires row, "
            "column and value."
        )

    wide = (
        data
        .pivot_table(
            index=row,
            columns=column,
            values=value,
            aggfunc="mean",
        )
        .reset_index()
    )

    value_columns = [
        current
        for current
        in wide.columns
        if current != row
    ]

    return {
        "data": wide,
        "row_label": row,
        "value_columns": value_columns,
        "title": parameters.get(
            "title"
        ),
    }


# ------------------------------------------------------------
# Dispatch one figure
# ------------------------------------------------------------

def dispatch_figure(
    data: pd.DataFrame,
    plan: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Path]:

    template_name = plan[
        "template"
    ]

    figure_id = plan[
        "figure"
    ]

    parameters = dict(
        plan[
            "parameters"
        ]
    )

    renderer = (
        TEMPLATE_REGISTRY.get(
            template_name
        )
    )

    if renderer is None:

        raise ValueError(
            f"Unknown template: "
            f"{template_name}"
        )

    output_dir = Path(
        output_dir
    )

    # --------------------------------------------------------
    # Line
    # --------------------------------------------------------

    if template_name == "line":

        kwargs = prepare_line_data(
            data,
            parameters,
            figure_id=figure_id,
        )

        kwargs[
            "output_dir"
        ] = output_dir

        return renderer(
            **kwargs
        )

    # --------------------------------------------------------
    # Grouped bar
    # --------------------------------------------------------

    if template_name == "grouped_bar":

        return renderer(
            data=data,
            category=parameters[
                "category"
            ],
            values=parameters[
                "values"
            ],
            title=parameters.get(
                "title"
            ),
            xlabel=parameters.get(
                "xlabel"
            ),
            ylabel=parameters.get(
                "ylabel"
            ),
            output_dir=output_dir,
        )

    # --------------------------------------------------------
    # Scatter / bubble
    # --------------------------------------------------------

    if template_name == "scatter_bubble":

        return renderer(
            data=data,
            x=parameters[
                "x"
            ],
            y=parameters[
                "y"
            ],
            label=parameters.get(
                "label"
            ),
            size=parameters.get(
                "size"
            ),
            highlight=parameters.get(
                "highlight"
            ),
            title=parameters.get(
                "title"
            ),
            xlabel=parameters.get(
                "xlabel"
            ),
            ylabel=parameters.get(
                "ylabel"
            ),
            output_dir=output_dir,
        )

    # --------------------------------------------------------
    # Violin + box
    # --------------------------------------------------------

    if template_name == "violin_box":

        return renderer(
            data=data,
            value=parameters.get(
                "value"
            ),
            group=parameters.get(
                "group"
            ),
            wide_columns=parameters.get(
                "wide_columns"
            ),
            title=parameters.get(
                "title"
            ),
            ylabel=parameters.get(
                "ylabel"
            ),
            output_dir=output_dir,
        )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    if template_name == "heatmap":

        kwargs = (
            prepare_heatmap_data(
                data,
                parameters,
            )
        )

        kwargs[
            "output_dir"
        ] = output_dir

        return renderer(
            **kwargs
        )

    # --------------------------------------------------------
    # Lollipop
    # --------------------------------------------------------

    if template_name == "lollipop":

        return renderer(
            data=data,
            category=parameters[
                "category"
            ],
            value=parameters[
                "value"
            ],
            highlight=parameters.get(
                "highlight"
            ),
            sort=parameters.get(
                "sort",
                True,
            ),
            title=parameters.get(
                "title"
            ),
            xlabel=parameters.get(
                "xlabel"
            ),
            output_dir=output_dir,
        )

    # --------------------------------------------------------
    # Stacked area
    # --------------------------------------------------------

    if template_name == "stacked_area":

        return renderer(
            data=data,
            x=parameters[
                "x"
            ],
            ys=parameters[
                "ys"
            ],
            normalize=parameters.get(
                "normalize",
                False,
            ),
            title=parameters.get(
                "title"
            ),
            xlabel=parameters.get(
                "xlabel"
            ),
            ylabel=parameters.get(
                "ylabel"
            ),
            output_dir=output_dir,
        )

    # --------------------------------------------------------
    # Ridge
    # --------------------------------------------------------

    if template_name == "ridge":

        return renderer(
            data=data,
            group=parameters[
                "group"
            ],
            value=parameters[
                "value"
            ],
            reference=parameters.get(
                "reference"
            ),
            title=parameters.get(
                "title"
            ),
            xlabel=parameters.get(
                "xlabel"
            ),
            output_dir=output_dir,
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    if template_name == "confusion_matrix":

        return renderer(
            matrix=parameters[
                "matrix"
            ],
            labels=parameters.get(
                "labels"
            ),
            normalize=parameters.get(
                "normalize",
                "row",
            ),
            title=parameters.get(
                "title"
            ),
            output_dir=output_dir,
        )

    # --------------------------------------------------------
    # Sankey
    # --------------------------------------------------------

    if template_name == "sankey":

        return renderer(
            data=data,
            source=parameters[
                "source"
            ],
            target=parameters[
                "target"
            ],
            value=parameters[
                "value"
            ],
            title=parameters.get(
                "title"
            ),
            output_dir=output_dir,
        )

    raise NotImplementedError(
        f"Dispatch logic not implemented "
        f"for template '{template_name}'."
    )


# ------------------------------------------------------------
# Dispatch all planned figures
# ------------------------------------------------------------

def dispatch_plan(
    data: pd.DataFrame,
    plan_report: dict[str, Any],
    *,
    output_root: str | Path = (
        "outputs/generated"
    ),
) -> dict[str, Any]:

    output_root = Path(
        output_root
    )

    generated = []
    failed = []

    for index, plan in enumerate(
        plan_report.get(
            "plans",
            [],
        ),
        start=1,
    ):

        figure_id = plan[
            "figure"
        ]

        current_dir = (
            output_root
            / (
                f"{index:02d}_"
                f"{figure_id}"
            )
        )

        try:

            paths = dispatch_figure(
                data,
                plan,
                output_dir=current_dir,
            )

        except Exception as exc:

            failed.append(
                {
                    "figure": figure_id,
                    "template": plan[
                        "template"
                    ],
                    "error": str(
                        exc
                    ),
                }
            )

            continue

        generated.append(
            {
                "figure": figure_id,
                "title": plan[
                    "title"
                ],
                "template": plan[
                    "template"
                ],
                "output_dir": str(
                    current_dir
                ),
                "files": {
                    key: str(
                        value
                    )
                    for key, value
                    in paths.items()
                },
            }
        )

    return {
        "generated": generated,
        "failed": failed,
    }


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

def print_dispatch_report(
    report: dict[str, Any],
) -> None:

    print(
        "sci-figure-maker dispatcher"
    )

    print(
        "---------------------------"
    )

    generated = report[
        "generated"
    ]

    if generated:

        print(
            "Generated figures:"
        )

        for index, item in enumerate(
            generated,
            start=1,
        ):

            print()

            print(
                f"{index}. "
                f"{item['title']}"
            )

            print(
                f"   Template: "
                f"{item['template']}"
            )

            print(
                f"   PNG: "
                f"{item['files']['png']}"
            )

            print(
                f"   SVG: "
                f"{item['files']['svg']}"
            )

            print(
                f"   PDF: "
                f"{item['files']['pdf']}"
            )

    else:

        print(
            "No figures were generated."
        )

    if report[
        "failed"
    ]:

        print()

        print(
            "Failed figures:"
        )

        for item in report[
            "failed"
        ]:

            print(
                f"  - "
                f"{item['figure']}: "
                f"{item['error']}"
            )


# ------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------

def parse_sheet(
    value: str,
) -> str | int:

    try:

        return int(
            value
        )

    except ValueError:

        return value


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-style "
            "figures from a scientific "
            "figure plan."
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
        "--output-dir",
        default=(
            "outputs/generated"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "outputs/generated/"
            "dispatch_report.json"
        ),
    )

    return parser


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

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

    report = dispatch_plan(
        data=data,
        plan_report=plan,
        output_root=args.output_dir,
    )

    print_dispatch_report(
        report
    )

    report_path = Path(
        args.report
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
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
        "Dispatch report saved to:"
    )

    print(
        f"  {report_path}"
    )


if __name__ == "__main__":
    main()