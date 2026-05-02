"""
pipeline.py
-----------
Orchestrates the full cleaning + transformation flow.

Design decisions:
  1. Steps are defined as an ordered list — easy to add/remove/reorder.
  2. Each step is timed and logged independently.
  3. Metadata (row counts, timing) is captured for observability.
  4. Chunked processing for files that don't fit in memory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from src import cleaners, transformers
from src.config import get_config
from src.logger import get_logger
from src.validators import DataValidator

log = get_logger(__name__)
cfg = get_config()


# ── Pipeline run metadata ─────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    input_rows: int = 0
    output_rows: int = 0
    input_cols: int = 0
    output_cols: int = 0
    steps_run: List[str] = field(default_factory=list)
    step_durations_ms: Dict[str, float] = field(default_factory=dict)
    validation_passed: bool = True
    errors: List[str] = field(default_factory=list)

    def log_summary(self) -> None:
        log.info("=" * 60)
        log.info("PIPELINE COMPLETE")
        log.info("  Rows : %d → %d (-%d)", self.input_rows, self.output_rows,
                 self.input_rows - self.output_rows)
        log.info("  Cols : %d → %d", self.input_cols, self.output_cols)
        log.info("  Steps: %s", ", ".join(self.steps_run))
        for step, ms in self.step_durations_ms.items():
            log.info("    %-40s %6.1f ms", step, ms)
        log.info("=" * 60)


# ── Pipeline class ────────────────────────────────────────────────────────────

class DataPipeline:
    """
    Composable data cleaning pipeline.

    Each step is a Callable[[pd.DataFrame], pd.DataFrame].
    Steps execute in order; any step can be swapped out or skipped.
    """

    # Default ordered step list — can be overridden at init
    DEFAULT_CLEANING_STEPS: List[Callable] = [
        cleaners.normalize_column_names,
        cleaners.drop_empty_columns,
        cleaners.drop_high_missing_columns,
        cleaners.drop_duplicates,
        cleaners.strip_string_columns,
        cleaners.normalise_email,
        cleaners.remove_pii_columns,
        cleaners.cast_column_types,
        cleaners.clip_outliers,
        cleaners.impute_missing,
    ]

    DEFAULT_TRANSFORM_STEPS: List[Callable] = [
        transformers.normalise_country_code,
        transformers.add_customer_tenure_days,
        transformers.add_age_bucket,
        transformers.flag_high_value_customers,
    ]

    def __init__(
        self,
        cleaning_steps: Optional[List[Callable]] = None,
        transform_steps: Optional[List[Callable]] = None,
        skip_validation: bool = False,
    ) -> None:
        self.cleaning_steps = cleaning_steps or self.DEFAULT_CLEANING_STEPS
        self.transform_steps = transform_steps or self.DEFAULT_TRANSFORM_STEPS
        self.skip_validation = skip_validation
        self.validator = DataValidator()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, PipelineResult]:
        result = PipelineResult(
            input_rows=len(df),
            input_cols=len(df.columns),
        )

        log.info("Pipeline '%s' started | shape=%s", cfg.pipeline.name, df.shape)

        # 1. Validate raw data
        if not self.skip_validation:
            report = self.validator.validate(df)
            result.validation_passed = report.passed
            result.errors = report.errors
            if not report.passed:
                log.error("Validation failed — aborting pipeline")
                return df, result

        # 2. Run cleaning steps
        df = self._run_steps(df, self.cleaning_steps, result)

        # 3. Run transformation steps
        df = self._run_steps(df, self.transform_steps, result)

        result.output_rows = len(df)
        result.output_cols = len(df.columns)
        result.log_summary()
        return df, result

    def run_file(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> PipelineResult:
        """
        Convenience method: reads a file, runs the pipeline, writes output.
        Supports CSV and Parquet. Falls back to chunked reading for large CSVs.
        """
        input_path = Path(input_path)
        output_path = Path(output_path) if output_path else (
            Path(cfg.data.processed_path) / f"clean_{input_path.name}"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("Reading %s", input_path)
        df = self._read_file(input_path)

        df, result = self.run(df)

        log.info("Writing output to %s", output_path)
        self._write_file(df, output_path)
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_steps(
        self,
        df: pd.DataFrame,
        steps: List[Callable],
        result: PipelineResult,
    ) -> pd.DataFrame:
        for step_fn in steps:
            name = step_fn.__name__
            t0 = time.perf_counter()
            try:
                df = step_fn(df)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                result.steps_run.append(name)
                result.step_durations_ms[name] = elapsed_ms
                log.debug("  ✓ %-40s %6.1f ms | shape=%s", name, elapsed_ms, df.shape)
            except Exception as exc:
                log.exception("Step '%s' failed: %s", name, exc)
                result.errors.append(f"{name}: {exc}")
                raise  # Re-raise — partial pipelines are dangerous
        return df

    def _read_file(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path, encoding=cfg.data.encoding, low_memory=False)
        if suffix == ".json":
            return pd.read_json(path, encoding=cfg.data.encoding)
        raise ValueError(f"Unsupported file format: {suffix}")

    def _write_file(self, df: pd.DataFrame, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df.to_parquet(path, index=False, engine="pyarrow")
        elif suffix == ".csv":
            df.to_csv(path, index=False, encoding=cfg.data.encoding)
        else:
            df.to_parquet(path.with_suffix(".parquet"), index=False)


    def _print_stats(self, df: pd.DataFrame) -> None:
        #print(f"[SUCCESS] CSV loaded successfully from {self.save_path}")
        print(f"[SHAPE], Dataset has this shape, {df.shape}")
        print(f"[COLUMNS], Dataset has these colums, {df.columns}")
        print(f"[INFO], Dataset has this info, {df.info()}")
        print(f"[STATS], Dataset has this statistical description, {df.describe()}")        
