"""
cleaners.py
-----------
Pure, stateless cleaning functions.
Each function takes a DataFrame, returns a DataFrame.
This makes them trivially testable and composable in a pipeline.

Convention:
  - Never mutate in-place (always work on a copy)
  - Log what was changed and how many rows/cols were affected
  - Raise ValueError for unrecoverable states
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import get_config
from src.logger import get_logger

log = get_logger(__name__)
cfg = get_config()


# ── Structural cleaners ───────────────────────────────────────────────────────

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase, strip whitespace, replace spaces/hyphens with underscores.
    'First Name' → 'first_name'
    """
    original = df.columns.tolist()
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    renamed = {o: n for o, n in zip(original, df.columns) if o != n}
    if renamed:
        log.info("Renamed %d column(s): %s", len(renamed), renamed)
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely null."""
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    dropped = before - df.shape[1]
    if dropped:
        log.info("Dropped %d fully-null column(s)", dropped)
    return df


def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Drop columns where the fraction of nulls exceeds `threshold`."""
    threshold = threshold or cfg.cleaning.missing_value_threshold
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    if cols_to_drop:
        log.info(
            "Dropping %d high-missing column(s) (>%.0f%%): %s",
            len(cols_to_drop), threshold * 100, cols_to_drop,
        )
        df = df.drop(columns=cols_to_drop)
    return df


def drop_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Drop duplicate rows, keeping the first occurrence."""
    subset = subset or cfg.cleaning.drop_duplicate_subset
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    dropped = before - len(df)
    if dropped:
        log.info("Dropped %d duplicate row(s)", dropped)
    return df


# ── Type-casting cleaners ─────────────────────────────────────────────────────

def cast_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to types defined in config schema.
    Rows that fail casting for a specific column are coerced to NaN (not dropped).
    """
    df = df.copy()
    type_map = cfg.schema_.column_types

    for col, dtype in type_map.items():
        if col not in df.columns:
            continue
        try:
            if "datetime" in dtype:
                df[col] = _parse_dates(df[col])
            elif dtype in ("int", "Int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "bool":
                df[col] = _parse_bool(df[col])
            else:
                df[col] = df[col].astype(dtype)
            log.debug("Cast '%s' → %s", col, dtype)
        except Exception as exc:
            log.warning("Could not cast '%s' to %s: %s", col, dtype, exc)

    return df


def _parse_dates(series: pd.Series) -> pd.Series:
    """Try multiple date formats before giving up."""
    for fmt in cfg.cleaning.date_formats:
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except (ValueError, TypeError):
            continue
    # Last resort: let pandas infer
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


def _parse_bool(series: pd.Series) -> pd.Series:
    """Map common truthy/falsy strings to bool."""
    true_vals = {"true", "yes", "1", "y", "t"}
    false_vals = {"false", "no", "0", "n", "f"}
    normalised = series.astype(str).str.strip().str.lower()
    result = pd.Series(np.nan, index=series.index, dtype=object)
    result[normalised.isin(true_vals)] = True
    result[normalised.isin(false_vals)] = False
    return result.astype("boolean")  # nullable boolean


# ── String cleaners ───────────────────────────────────────────────────────────

def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns."""
    df = df.copy()
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    log.debug("Stripped whitespace from %d string column(s)", len(str_cols))
    return df


def normalise_email(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase emails and remove whitespace."""
    if "email" not in df.columns:
        return df
    df = df.copy()
    df["email"] = df["email"].str.lower().str.strip()
    return df


def remove_pii_columns(
    df: pd.DataFrame,
    pii_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Drop PII columns before writing to data warehouse.
    Default list can be extended via argument.
    """
    defaults = ["ssn", "social_security", "password", "credit_card", "ip_address"]
    to_drop = list(set(defaults + (pii_columns or [])))
    existing = [c for c in to_drop if c in df.columns]
    if existing:
        log.info("Removing PII column(s): %s", existing)
        df = df.drop(columns=existing)
    return df


# ── Numeric cleaners ──────────────────────────────────────────────────────────

def clip_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_std: Optional[float] = None,
) -> pd.DataFrame:
    """
    Clip values beyond N standard deviations to the boundary value.
    Clipping is preferred over dropping in most production pipelines
    because it preserves row count.
    """
    n_std = n_std or cfg.cleaning.outlier_std_threshold
    df = df.copy()
    num_cols = columns or df.select_dtypes(include="number").columns.tolist()

    for col in num_cols:
        series = df[col].dropna()
        mean, std = series.mean(), series.std()
        if std == 0:
            continue
        lower, upper = mean - n_std * std, mean + n_std * std
        clipped = df[col].clip(lower=lower, upper=upper)
        n_changed = (clipped != df[col]).sum()
        if n_changed:
            log.info("Clipped %d outlier(s) in '%s'", n_changed, col)
        df[col] = clipped

    return df


def impute_missing(
    df: pd.DataFrame,
    strategy: str = "median",  # 'median' | 'mean' | 'mode' | 'constant'
    fill_value=None,
) -> pd.DataFrame:
    """
    Impute missing values in numeric columns.
    Categorical columns are filled with the mode unless fill_value provided.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    for col in num_cols:
        if df[col].isnull().sum() == 0:
            continue
        if strategy == "median":
            val = df[col].median()
        elif strategy == "mean":
            val = df[col].mean()
        elif strategy == "constant":
            val = fill_value if fill_value is not None else 0
        else:
            val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
        df[col] = df[col].fillna(val)
        log.debug("Imputed '%s' with %s=%.4f", col, strategy, val)

    for col in cat_cols:
        if df[col].isnull().sum() == 0:
            continue
        val = fill_value if fill_value is not None else (
            df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
        )
        df[col] = df[col].fillna(val)
        log.debug("Imputed categorical '%s' with '%s'", col, val)

    return df
