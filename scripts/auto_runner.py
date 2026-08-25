from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.figure_dispatch import (
    dispatch_plan,
)
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


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def parse_sheet(
    value: str,
) -> str | int:
    """
    Excel sheet can be either:

    --sheet 0

    or:

    --sheet Benchmark
    """

    try:
        return int(value)

    except ValueError:
        return value


def save_json(
    data: Any,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


# ------------------------------------------------------------
# Automatic pipeline
# ------------------------------------------------------------

def run_pipeline(
    input_path: str | Path,
    *,
    sheet: str | int = 0,
    highlight: str | None = None,
    top: int = 3,
    output_root: str | Path = "outputs/auto",
) -> dict[str, Any]:
    """
    Run the complete sci-figure-maker MVP pipeline:

    input file
        ↓
    load data
        ↓
    inspect scientific semantics
        ↓
    recommend figures
        ↓
    plan plotting parameters
        ↓
    dispatch visual templates
        ↓
    export PNG / SVG / PDF
    """

    input_path = Path(
        input_path
    )

    output_root = Path(
        output_root
    )

    # Each experiment receives its own directory.
    run_dir = (
        output_root
        / input_path.stem
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_dir = (
        run_dir
        / "figures"
    )

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    data = load_data(
        input_path,
        sheet_name=sheet,
    )

    # --------------------------------------------------------
    # 2. Inspect
    # --------------------------------------------------------

    inspection = inspect_data(
        data,
        source=input_path,
    )

    # --------------------------------------------------------
    # 3. Select
    # --------------------------------------------------------

    selection = (
        build_selection_report(
            inspection
        )
    )

    # --------------------------------------------------------
    # 4. Plan
    # --------------------------------------------------------

    plan = build_figure_plan(
        data=data,
        inspection=inspection,
        selection=selection,
        highlight=highlight,
        top=top,
    )

    # --------------------------------------------------------
    # 5. Dispatch / render
    # --------------------------------------------------------

    dispatch = dispatch_plan(
        data=data,
        plan_report=plan,
        output_root=figure_dir,
    )

    # --------------------------------------------------------
    # 6. Save intermediate reports
    # --------------------------------------------------------

    save_json(
        inspection,
        run_dir
        / "inspection.json",
    )

    save_json(
        selection,
        run_dir
        / "selection.json",
    )

    save_json(
        plan,
        run_dir
        / "figure_plan.json",
    )

    save_json(
        dispatch,
        run_dir
        / "dispatch.json",
    )

    # --------------------------------------------------------
    # 7. Build master run report
    # --------------------------------------------------------

    report = {
        "source": str(
            input_path
        ),
        "sheet": sheet,
        "highlight": highlight,
        "requested_top": top,
        "output_directory": str(
            run_dir
        ),
        "data_shape": {
            "rows": int(
                len(data)
            ),
            "columns": int(
                len(data.columns)
            ),
        },
        "candidate_experiments": (
            inspection.get(
                "candidate_experiments",
                [],
            )
        ),
        "recommendations": (
            selection.get(
                "recommendations",
                [],
            )
        ),
        "figure_plans": (
            plan.get(
                "plans",
                [],
            )
        ),
        "generated": (
            dispatch.get(
                "generated",
                [],
            )
        ),
        "failed": (
            dispatch.get(
                "failed",
                [],
            )
        ),
        "warnings": (
            inspection.get(
                "warnings",
                [],
            )
        ),
    }

    save_json(
        report,
        run_dir
        / "run_report.json",
    )

    return report


# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------

def print_run_report(
    report: dict[str, Any],
) -> None:

    print()
    print(
        "sci-figure-maker automatic pipeline"
    )

    print(
        "==================================="
    )

    print(
        f"Source: {report['source']}"
    )

    print(
        "Data:",
        f"{report['data_shape']['rows']} rows x",
        f"{report['data_shape']['columns']} columns",
    )

    print()

    experiments = report[
        "candidate_experiments"
    ]

    if experiments:

        print(
            "Detected experiment types:"
        )

        for item in experiments:

            print(
                f"  - {item['type']} "
                f"({item['confidence']})"
            )

    else:

        print(
            "No known experiment type "
            "was confidently detected."
        )

    print()

    generated = report[
        "generated"
    ]

    if generated:

        print(
            "Generated figure candidates:"
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
                f"   Figure: "
                f"{item['figure']}"
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
            "No figure candidates "
            "were generated."
        )

    if report["failed"]:

        print()

        print("Failed:")

        for item in report[
            "failed"
        ]:

            print(
                f"  - {item['figure']}: "
                f"{item['error']}"
            )

    if report["warnings"]:

        print()

        print("Warnings:")

        for warning in report[
            "warnings"
        ]:

            print(
                f"  - {warning}"
            )

    print()

    print(
        "Complete run saved to:"
    )

    print(
        f"  {report['output_directory']}"
    )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Automatically inspect scientific "
            "experimental data, recommend suitable "
            "figures and generate publication-style "
            "figure candidates."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input CSV, TSV, TXT "
            "or XLSX file."
        ),
    )

    parser.add_argument(
        "--sheet",
        default="0",
        help=(
            "Excel sheet name or "
            "zero-based sheet index."
        ),
    )

    parser.add_argument(
        "--highlight",
        default=None,
        help=(
            "Optional model / method / "
            "variant to emphasize."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help=(
            "Maximum number of figure "
            "candidates to generate."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/auto",
        help=(
            "Root directory for "
            "automatic runs."
        ),
    )

    return parser


def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    report = run_pipeline(
        input_path=args.input,
        sheet=parse_sheet(
            str(args.sheet)
        ),
        highlight=args.highlight,
        top=args.top,
        output_root=args.output_dir,
    )

    print_run_report(
        report
    )


if __name__ == "__main__":
    main()