"""
transformers.py
---------------
Business-logic transformations that run AFTER cleaning.
Cleaners make data consistent; transformers add value.

These are also pure functions (df in → df out).
"""

from __future__ import annotations

import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)


def add_customer_tenure_days(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate days since signup as of the pipeline run date."""
    if "signup_date" not in df.columns:
        return df
    df = df.copy()
    df["tenure_days"] = (pd.Timestamp.now() - df["signup_date"]).dt.days
    log.info("Added 'tenure_days' column")
    return df


def add_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bin continuous age into labelled brackets."""
    if "age" not in df.columns:
        return df
    df = df.copy()
    bins = [0, 18, 25, 35, 50, 65, 120]
    labels = ["<18", "18-24", "25-34", "35-49", "50-64", "65+"]
    df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    log.info("Added 'age_bucket' column")
    return df


def normalise_country_code(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase and strip country codes."""
    if "country" not in df.columns:
        return df
    df = df.copy()
    df["country"] = df["country"].str.upper().str.strip()
    return df


def flag_high_value_customers(
    df: pd.DataFrame,
    revenue_col: str = "revenue",
    percentile: float = 0.9,
) -> pd.DataFrame:
    """Binary flag for customers in the top N-th revenue percentile."""
    if revenue_col not in df.columns:
        return df
    df = df.copy()
    threshold = df[revenue_col].quantile(percentile)
    df["is_high_value"] = df[revenue_col] >= threshold
    n_flagged = df["is_high_value"].sum()
    log.info(
        "Flagged %d high-value customers (revenue >= %.2f, p%.0f)",
        n_flagged, threshold, percentile * 100,
    )
    return df
