"""
main.py
-------
CLI entrypoint. In production this is triggered by Airflow / Prefect / cron.

Usage:
    python main.py --input data/raw/customers.csv
    python main.py --input data/raw/customers.csv --output data/processed/clean.parquet
    python main.py --input data/raw/customers.csv --skip-validation
"""

import argparse
import sys
from pathlib import Path

from src.logger import get_logger
from src.pipeline import DataPipeline

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer data cleaning pipeline")
    parser.add_argument("--input", required=True, help="Path to raw input file")
    parser.add_argument("--output", default=None, help="Path for cleaned output")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip schema/quality validation (use with care)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        return 1

    pipeline = DataPipeline(skip_validation=args.skip_validation)

    try:
        result = pipeline.run_file(
            input_path=input_path,
            output_path=args.output,
        )
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        return 1

    if not result.validation_passed:
        log.error("Pipeline aborted due to validation errors")
        return 1

    log.info(
        "Done. %d → %d rows, %d → %d columns",
        result.input_rows, result.output_rows,
        result.input_cols, result.output_cols,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
