from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.figure_planner import (
    build_figure_plan,
)
from scripts.inspect_data import (
    inspect_data,
)
from scripts.load_data import (
    load_data,
)
from scripts.select_plot import (
    build_selection_report,
)
from scripts.template_registry import (
    TEMPLATE_REGISTRY,
)


# ------------------------------------------------------------
# Template preparation helpers
# ------------------------------------------------------------

def prepare_line_data(
    data: pd.DataFrame,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a generic curve plan into arguments accepted by
    templates.line.render_line().
    """

    x = parameters.get("x")
    y = parameters.get("y")
    ys = parameters.get("ys")
    group = parameters.get("group")

    # Standard multi-column training curve:
    #
    # Epoch | train_loss | val_loss | mAP...
    if ys:

        return {
            "data": data,
            "x": x,
            "ys": ys,
            "title": parameters.get("title"),
            "xlabel": parameters.get("xlabel"),
            "ylabel": parameters.get("ylabel"),
        }

    # Long-form grouped curve:
    #
    # Model | Recall | Precision
    #
    # needs to be converted into wide format so that
    # each model becomes one line.
    if (
        group
        and x
        and y
    ):

        working = data[
            [group, x, y]
        ].copy()

        working = working.dropna(
            subset=[
                group,
                x,
                y,
            ]
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

        return {
            "data": wide,
            "x": x,
            "ys": curve_columns,
            "labels": [
                str(column)
                for column
                in curve_columns
            ],
            "title": parameters.get("title"),
            "xlabel": parameters.get("xlabel"),
            "ylabel": parameters.get("ylabel"),
        }

    # Single curve:
    #
    # Recall | Precision
    if (
        x
        and y
    ):

        return {
            "data": data,
            "x": x,
            "ys": [y],
            "title": parameters.get("title"),
            "xlabel": parameters.get("xlabel"),
            "ylabel": parameters.get("ylabel"),
        }

    raise ValueError(
        "Unable to prepare line-plot data."
    )


def prepare_heatmap_data(
    data: pd.DataFrame,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert long-form robustness data:

    Model | Scenario | Metric

    into wide-form:

    Model | Smoke | Blur | ...
    """

    row = parameters.get("row")
    column = parameters.get("column")
    value = parameters.get("value")

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

    parameters = dict(
        plan["parameters"]
    )

    renderer = TEMPLATE_REGISTRY.get(
        template_name
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
        )

        kwargs["output_dir"] = (
            output_dir
        )

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
            x=parameters["x"],
            y=parameters["y"],
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

        kwargs = prepare_heatmap_data(
            data,
            parameters,
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
    output_root: str | Path = "outputs/generated",
) -> dict[str, Any]:

    output_root = Path(
        output_root
    )

    generated = []
    failed = []

    for index, plan in enumerate(
        plan_report.get(
            "plans",
            []
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
                    "error": str(exc),
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
                    key: str(value)
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
# Console output
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

    if report["failed"]:

        print()

        print("Failed figures:")

        for item in report[
            "failed"
        ]:

            print(
                f"  - {item['figure']}: "
                f"{item['error']}"
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
        default="outputs/generated",
    )

    parser.add_argument(
        "--report",
        default=(
            "outputs/generated/"
            "dispatch_report.json"
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