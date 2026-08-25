from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
}


def load_data(
    path: str | Path,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """
    Load scientific tabular data from a supported file.

    Supported formats
    -----------------
    .csv
        Comma-separated values.

    .tsv
        Tab-separated values.

    .txt
        Treated as tab-separated text by default.

    .xlsx
        Microsoft Excel workbook.

    Parameters
    ----------
    path:
        Input file path.

    sheet_name:
        Excel sheet name or sheet index.
        Ignored for non-Excel files.

    Returns
    -------
    pandas.DataFrame
        Loaded table.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.

    ValueError
        If the format is unsupported or the loaded
        table contains no rows or columns.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(
            sorted(SUPPORTED_EXTENSIONS)
        )

        raise ValueError(
            f"Unsupported file format: '{extension}'. "
            f"Supported formats: {supported}"
        )

    if extension == ".csv":

        data = pd.read_csv(path)

    elif extension == ".tsv":

        data = pd.read_csv(
            path,
            sep="\t",
        )

    elif extension == ".txt":

        data = pd.read_csv(
            path,
            sep="\t",
        )

    elif extension == ".xlsx":

        data = pd.read_excel(
            path,
            sheet_name=sheet_name,
            engine="openpyxl",
        )

    else:
        # This branch should normally never be reached,
        # because unsupported extensions are rejected above.
        raise ValueError(
            f"Cannot load file: {path}"
        )

    if data.empty:
        raise ValueError(
            f"The input file contains no data rows: {path}"
        )

    if len(data.columns) == 0:
        raise ValueError(
            f"The input file contains no columns: {path}"
        )

    return data


def summarize_data(
    data: pd.DataFrame,
) -> dict:
    """
    Produce a lightweight summary of the loaded dataset.
    """

    summary = {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "column_names": list(data.columns),
        "missing_values": {
            column: int(data[column].isna().sum())
            for column in data.columns
        },
        "dtypes": {
            column: str(data[column].dtype)
            for column in data.columns
        },
    }

    return summary


def print_summary(
    data: pd.DataFrame,
    source: Path,
) -> None:
    """
    Print a human-readable dataset summary.
    """

    summary = summarize_data(data)

    print("sci-figure-maker data loader")
    print("----------------------------")

    print(f"Source: {source}")
    print(f"Rows: {summary['rows']}")
    print(f"Columns: {summary['columns']}")

    print()
    print("Column information:")

    for column in summary["column_names"]:

        dtype = summary["dtypes"][column]
        missing = summary["missing_values"][column]

        print(
            f"  - {column}: "
            f"dtype={dtype}, "
            f"missing={missing}"
        )

    print()
    print("Preview:")
    print(data.head())


def build_parser() -> argparse.ArgumentParser:
    """
    Build command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load and inspect scientific tabular data."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input CSV, TSV, TXT, or XLSX file."
        ),
    )

    parser.add_argument(
        "--sheet",
        default=0,
        help=(
            "Excel sheet name or zero-based sheet index. "
            "Ignored for non-Excel files."
        ),
    )

    return parser


def parse_sheet_argument(
    value: str,
) -> str | int:
    """
    Convert a numeric sheet argument to an integer.
    Keep text sheet names as strings.
    """

    try:
        return int(value)

    except ValueError:
        return value


def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    input_path = Path(args.input)

    sheet_name = parse_sheet_argument(
        str(args.sheet)
    )

    data = load_data(
        input_path,
        sheet_name=sheet_name,
    )

    print_summary(
        data,
        source=input_path,
    )


if __name__ == "__main__":
    main()