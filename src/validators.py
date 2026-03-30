"""
validators.py
-------------
Validate raw DataFrames BEFORE cleaning begins.
Returns a structured ValidationReport — never raises silently.

Pattern: fail fast on schema issues, warn on data quality issues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.config import get_config
from src.logger import get_logger

log = get_logger(__name__)


# ── Report model ──────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    passed: bool = True
    errors: List[str] = field(default_factory=list)   # blocking
    warnings: List[str] = field(default_factory=list) # non-blocking

    def add_error(self, msg: str) -> None:
        log.error("VALIDATION ERROR: %s", msg)
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        log.warning("VALIDATION WARN:  %s", msg)
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status} | "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )


# ── Validator class ───────────────────────────────────────────────────────────

class DataValidator:
    """
    Runs schema + business-rule checks on a raw DataFrame.

    Usage:
        report = DataValidator().validate(df)
        if not report.passed:
            raise ValueError(report.summary())
    """

    EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$")

    def __init__(self) -> None:
        self.cfg = get_config()

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        log.info("Starting validation on DataFrame shape=%s", df.shape)
        report = ValidationReport()

        self._check_not_empty(df, report)
        if not report.passed:            # no point continuing on empty frame
            return report

        self._check_required_columns(df, report)
        self._check_duplicate_rows(df, report)
        self._check_missing_rate(df, report)
        self._check_email_format(df, report)
        self._check_numeric_ranges(df, report)

        log.info(report.summary())
        return report

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_not_empty(self, df: pd.DataFrame, r: ValidationReport) -> None:
        if df.empty:
            r.add_error("DataFrame is empty — nothing to process")

    def _check_required_columns(self, df: pd.DataFrame, r: ValidationReport) -> None:
        required = set(self.cfg.schema_.required_columns)
        missing = required - set(df.columns)
        if missing:
            r.add_error(f"Missing required columns: {sorted(missing)}")

    def _check_duplicate_rows(self, df: pd.DataFrame, r: ValidationReport) -> None:
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            pct = n_dupes / len(df) * 100
            r.add_warning(f"{n_dupes} duplicate rows ({pct:.1f}%) — will be dropped")

    def _check_missing_rate(self, df: pd.DataFrame, r: ValidationReport) -> None:
        threshold = self.cfg.cleaning.missing_value_threshold
        rates: Dict[str, float] = (df.isnull().mean()).to_dict()
        bad_cols = {col: rate for col, rate in rates.items() if rate > threshold}
        for col, rate in bad_cols.items():
            r.add_warning(
                f"Column '{col}' has {rate*100:.1f}% missing values "
                f"(threshold={threshold*100:.0f}%) — will be dropped"
            )

    def _check_email_format(self, df: pd.DataFrame, r: ValidationReport) -> None:
        if "email" not in df.columns:
            return
        emails = df["email"].dropna().astype(str)
        invalid_mask = ~emails.str.match(self.EMAIL_RE)
        n_invalid = invalid_mask.sum()
        if n_invalid:
            examples = emails[invalid_mask].head(3).tolist()
            r.add_warning(
                f"{n_invalid} invalid email(s) found. Examples: {examples}"
            )

    def _check_numeric_ranges(self, df: pd.DataFrame, r: ValidationReport) -> None:
        """Flag columns whose values exceed ± N standard deviations."""
        threshold = self.cfg.cleaning.outlier_std_threshold
        for col in df.select_dtypes(include="number").columns:
            series = df[col].dropna()
            if series.empty:
                continue
            mean, std = series.mean(), series.std()
            if std == 0:
                continue
            outliers = ((series - mean).abs() > threshold * std).sum()
            if outliers:
                r.add_warning(
                    f"Column '{col}': {outliers} outlier(s) "
                    f"beyond {threshold}σ (mean={mean:.2f}, std={std:.2f})"
                )
