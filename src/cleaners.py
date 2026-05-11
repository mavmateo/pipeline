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

import hashlib
import json
import re
import unicodedata
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from src.config import get_config
from src.logger import get_logger

log = get_logger(__name__)
cfg = get_config()


# ── Structural cleaners ───────────────────────────────────────────────────────

METADATA_PREFIX = "_"


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


def deduplicate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure column names are unique after normalization/aliasing.
    Duplicate names receive a stable numeric suffix.
    """
    seen: dict[str, int] = {}
    renamed: dict[str, str] = {}
    new_columns: list[str] = []

    for col in df.columns:
        count = seen.get(col, 0) + 1
        seen[col] = count
        new_col = col if count == 1 else f"{col}_{count}"
        new_columns.append(new_col)
        if new_col != col:
            renamed[col] = new_col

    if not renamed:
        return df

    df = df.copy()
    df.columns = new_columns
    log.info("Deduplicated %d duplicate column name(s): %s", len(renamed), renamed)
    return df


def apply_column_aliases(
    df: pd.DataFrame,
    aliases: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Rename source-specific column aliases to canonical schema names.
    Config aliases are expected after column normalization.
    """
    aliases = aliases or cfg.cleaning.column_aliases
    if not aliases:
        return df

    existing = {source: target for source, target in aliases.items() if source in df.columns}
    if not existing:
        return df

    df = df.rename(columns=existing)
    log.info("Applied %d column alias(es): %s", len(existing), existing)
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
    if subset is None:
        subset = [col for col in df.columns if not str(col).startswith(METADATA_PREFIX)]
    if not subset:
        return df
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


# ── Source and text standardisation ───────────────────────────────────────────

def normalize_unicode_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Unicode text so visually similar values compare consistently."""
    df = df.copy()
    text_cols = _text_columns(df)

    for col in text_cols:
        df[col] = df[col].map(
            lambda value: unicodedata.normalize("NFKC", value)
            if isinstance(value, str)
            else value
        )

    if text_cols:
        log.debug("Normalized Unicode text in %d column(s)", len(text_cols))
    return df


def clean_control_characters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-printing control characters from text columns."""
    df = df.copy()
    text_cols = _text_columns(df)
    control_re = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")

    for col in text_cols:
        df[col] = df[col].map(
            lambda value: control_re.sub(" ", value) if isinstance(value, str) else value
        )

    if text_cols:
        log.debug("Removed control characters from %d column(s)", len(text_cols))
    return df


def standardize_null_values(
    df: pd.DataFrame,
    null_values: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert common source null tokens to pandas missing values."""
    null_values = null_values or cfg.cleaning.null_values
    normalized_nulls = {
        str(value).strip().lower()
        for value in null_values
        if value is not None
    }
    if not normalized_nulls:
        return df

    df = df.copy()
    text_cols = _text_columns(df)
    changed = 0

    for col in text_cols:
        normalized = df[col].astype("string").str.strip().str.lower()
        mask = normalized.isin(normalized_nulls)
        changed += int(mask.sum())
        df.loc[mask, col] = pd.NA

    if changed:
        log.info("Standardized %d source null value(s)", changed)
    return df


def add_ingestion_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add source lineage and row-level audit metadata."""
    df = df.copy()
    source_file = df.attrs.get("source_file", "in_memory")
    ingested_at = pd.Timestamp.now(tz="UTC")

    df["_source_file"] = source_file
    df["_source_system"] = cfg.pipeline.name
    df["_ingested_at"] = ingested_at
    df["_row_number"] = np.arange(1, len(df) + 1)
    df["_row_hash"] = _row_hashes(df.drop(columns=[
        "_source_file",
        "_source_system",
        "_ingested_at",
        "_row_number",
    ]))

    log.debug("Added ingestion metadata columns")
    return df


# ── String cleaners ───────────────────────────────────────────────────────────

def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns."""
    df = df.copy()
    str_cols = _text_columns(df)
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


def standardize_categorical_values(
    df: pd.DataFrame,
    value_maps: Optional[dict[str, dict[str, str]]] = None,
) -> pd.DataFrame:
    """Map source-specific category labels to canonical values."""
    value_maps = value_maps or cfg.cleaning.categorical_value_maps
    if not value_maps:
        return df

    df = df.copy()
    for col, mapping in value_maps.items():
        if col not in df.columns:
            continue

        normalized_mapping = {
            str(source).strip().lower(): target
            for source, target in mapping.items()
        }
        normalized_values = df[col].astype("string").str.strip().str.lower()
        mapped = normalized_values.map(normalized_mapping)
        mask = mapped.notna()
        if mask.any():
            df.loc[mask, col] = mapped[mask]
            log.info("Standardized %d categorical value(s) in '%s'", int(mask.sum()), col)

    return df


def normalize_phone_numbers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Normalize phone-like columns to a simple E.164-style format."""
    columns = columns or cfg.cleaning.phone_columns
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return df

    df = df.copy()
    for col in existing:
        df[col] = df[col].map(_normalize_phone_value)
        log.info("Normalized phone numbers in '%s'", col)
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

def normalize_currency_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Parse currency strings into numeric values."""
    columns = columns or cfg.cleaning.currency_columns
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return df

    df = df.copy()
    for col in existing:
        df[col] = df[col].map(_parse_currency_value)
        log.info("Normalized currency values in '%s'", col)
    return df


def validate_numeric_ranges(
    df: pd.DataFrame,
    ranges: Optional[dict[str, dict[str, Optional[float]]]] = None,
) -> pd.DataFrame:
    """
    Replace configured out-of-range numeric values with missing values.
    The imputation step can then fill these values explicitly.
    """
    ranges = ranges or cfg.cleaning.numeric_ranges
    if not ranges:
        return df

    df = df.copy()
    for col, bounds in ranges.items():
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        mask = pd.Series(False, index=df.index)
        min_value = bounds.get("min")
        max_value = bounds.get("max")

        if min_value is not None:
            mask |= numeric < min_value
        if max_value is not None:
            mask |= numeric > max_value

        n_invalid = int(mask.sum())
        if n_invalid:
            df.loc[mask, col] = pd.NA
            log.info("Set %d out-of-range value(s) in '%s' to null", n_invalid, col)

    return df


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
        if pd.isna(std) or std == 0:
            continue
        lower, upper = mean - n_std * std, mean + n_std * std
        clipped = df[col].clip(lower=lower, upper=upper)
        n_changed = (clipped != df[col]).fillna(False).sum()
        if n_changed:
            log.info("Clipped %d outlier(s) in '%s'", n_changed, col)
        df[col] = clipped

    return df


def add_missingness_flags(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add boolean indicator columns for values that are missing before imputation."""
    df = df.copy()
    columns = columns or [
        col for col in df.columns
        if not str(col).startswith(METADATA_PREFIX) and df[col].isnull().any()
    ]

    added = 0
    for col in columns:
        if col not in df.columns or str(col).startswith(METADATA_PREFIX):
            continue
        flag_col = f"{col}_was_missing"
        if flag_col in df.columns:
            continue
        df[flag_col] = df[col].isnull()
        added += 1

    if added:
        log.info("Added %d missingness flag column(s)", added)
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


# ── Complex conformance cleaners ──────────────────────────────────────────────

def parse_nested_json_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Expand JSON object columns into top-level columns using a source prefix."""
    columns = columns or cfg.cleaning.json_columns or _detect_json_columns(df)
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return df

    df = df.copy()
    for col in existing:
        parsed = df[col].map(_parse_json_object)
        dict_rows = parsed[parsed.map(lambda value: isinstance(value, dict))]
        if dict_rows.empty:
            continue

        expanded = pd.json_normalize(dict_rows).set_index(dict_rows.index)
        expanded.columns = [f"{col}_{_normalize_generated_column_name(name)}" for name in expanded.columns]
        df = df.join(expanded, how="left")
        log.info("Expanded JSON column '%s' into %d column(s)", col, len(expanded.columns))

    return df


def validate_and_quarantine_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove blocking-invalid rows and store them in df.attrs['quarantine_rows'].
    Currently blocks rows missing configured required columns.
    """
    required = [col for col in cfg.schema_.required_columns if col in df.columns]
    if not required:
        return df

    invalid_reasons = pd.Series("", index=df.index, dtype="string")
    for col in required:
        missing = df[col].isnull()
        invalid_reasons.loc[missing] = (
            invalid_reasons.loc[missing]
            .str.cat(pd.Series([f"missing {col}"] * int(missing.sum()), index=df.index[missing]), sep="; ")
            .str.strip("; ")
        )

    invalid_mask = invalid_reasons.ne("")
    if not invalid_mask.any():
        return df

    valid_df = df.loc[~invalid_mask].copy()
    quarantine = df.loc[invalid_mask].copy()
    quarantine["_quarantine_reason"] = invalid_reasons.loc[invalid_mask]
    valid_df.attrs.update(df.attrs)
    existing_quarantine = df.attrs.get("quarantine_rows")
    if isinstance(existing_quarantine, pd.DataFrame):
        quarantine = pd.concat([existing_quarantine, quarantine], ignore_index=True)
    valid_df.attrs["quarantine_rows"] = quarantine

    log.warning("Quarantined %d row(s) with blocking quality issues", len(quarantine))
    return valid_df


def resolve_fuzzy_duplicates(
    df: pd.DataFrame,
    key_groups: Optional[List[List[str]]] = None,
) -> pd.DataFrame:
    """
    Resolve duplicate business entities using configured normalized key groups.
    Keeps the row with the highest non-null completeness for each matching key.
    """
    key_groups = key_groups or cfg.cleaning.fuzzy_duplicate_keys
    if not key_groups:
        return df

    result = df.copy()
    for keys in key_groups:
        existing = [key for key in keys if key in result.columns]
        if len(existing) != len(keys):
            continue

        key_series = _normalized_composite_key(result, existing)
        populated = key_series.ne("")
        if not populated.any():
            continue

        before = len(result)
        completeness = result.notna().sum(axis=1)
        ordered = result.assign(
            _dedupe_key=key_series,
            _completeness=completeness,
        ).sort_values("_completeness", ascending=False)
        ordered = ordered.drop_duplicates(subset=["_dedupe_key"], keep="first")
        ordered = ordered.sort_index()
        result = ordered.drop(columns=["_dedupe_key", "_completeness"])
        dropped = before - len(result)
        if dropped:
            log.info("Resolved %d fuzzy duplicate row(s) using keys %s", dropped, keys)

    return result


def normalize_dates_and_timezones(
    df: pd.DataFrame,
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """Convert configured datetime columns to a consistent timezone-aware dtype."""
    timezone = timezone or cfg.cleaning.datetime_timezone
    datetime_cols = [
        col for col, dtype in cfg.schema_.column_types.items()
        if col in df.columns and "datetime" in dtype
    ]
    if not datetime_cols:
        return df

    df = df.copy()
    for col in datetime_cols:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        if timezone and timezone.upper() != "UTC":
            parsed = parsed.dt.tz_convert(timezone)
        df[col] = parsed
        log.debug("Normalized datetime timezone for '%s'", col)
    return df


def _text_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "string"]).columns.tolist()


def _row_hashes(df: pd.DataFrame) -> pd.Series:
    def hash_row(row: pd.Series) -> str:
        payload = json.dumps(row.where(pd.notna(row), None).to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return df.apply(hash_row, axis=1)


def _normalize_phone_value(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA

    has_plus = text.startswith("+")
    digits = re.sub(r"\D", "", text)
    if not digits:
        return pd.NA
    if has_plus:
        return f"+{digits}"
    if len(digits) == 10:
        return f"{cfg.cleaning.default_phone_country_code}{digits}"
    if len(digits) > 10:
        return f"+{digits}"
    return digits


def _parse_currency_value(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, float, np.number)):
        return value

    text = str(value).strip().lower()
    if not text:
        return pd.NA

    is_negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    multiplier = 1.0
    if text.endswith("k"):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("b"):
        multiplier = 1_000_000_000.0
        text = text[:-1]

    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", "-", ".", "-."}:
        return pd.NA

    try:
        parsed = float(text) * multiplier
    except ValueError:
        return pd.NA
    return -parsed if is_negative else parsed


def _detect_json_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for col in _text_columns(df):
        sample = df[col].dropna().astype(str).str.strip().head(20)
        if sample.empty:
            continue
        if sample.str.startswith("{").any():
            candidates.append(col)
    return candidates


def _parse_json_object(value: Any) -> Any:
    if pd.isna(value) or isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text.startswith("{"):
        return value
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return value
    return parsed if isinstance(parsed, dict) else value


def _normalize_generated_column_name(name: Any) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(".", "_")
        .replace(" ", "_")
    )


def _normalized_composite_key(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    parts = []
    for key in keys:
        part = (
            df[key]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
        parts.append(part)
    return pd.concat(parts, axis=1).agg("|".join, axis=1).str.strip("|")
