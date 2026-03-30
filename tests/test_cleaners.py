"""
test_cleaners.py
----------------
Unit tests for cleaning functions.
Each test:
  - Creates a minimal DataFrame with only the data needed
  - Calls exactly one function
  - Asserts precise outputs

Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.cleaners import (
    clip_outliers,
    drop_duplicates,
    drop_high_missing_columns,
    impute_missing,
    normalize_column_names,
    normalise_email,
    strip_string_columns,
)


# ── normalize_column_names ────────────────────────────────────────────────────

class TestNormalizeColumnNames:
    def test_lowercase(self):
        df = pd.DataFrame(columns=["First Name", "Last Name"])
        result = normalize_column_names(df)
        assert list(result.columns) == ["first_name", "last_name"]

    def test_spaces_to_underscores(self):
        df = pd.DataFrame(columns=["customer id"])
        result = normalize_column_names(df)
        assert "customer_id" in result.columns

    def test_hyphens_to_underscores(self):
        df = pd.DataFrame(columns=["sign-up-date"])
        result = normalize_column_names(df)
        assert "sign_up_date" in result.columns  # hyphens → underscores, spaces not collapsed

    def test_already_clean_columns_unchanged(self):
        df = pd.DataFrame(columns=["email", "age"])
        result = normalize_column_names(df)
        assert list(result.columns) == ["email", "age"]


# ── drop_high_missing_columns ─────────────────────────────────────────────────

class TestDropHighMissingColumns:
    def test_drops_column_above_threshold(self):
        df = pd.DataFrame({
            "keep": [1, 2, 3, 4],
            "drop_me": [None, None, None, 1],  # 75% missing
        })
        result = drop_high_missing_columns(df, threshold=0.5)
        assert "keep" in result.columns
        assert "drop_me" not in result.columns

    def test_keeps_column_below_threshold(self):
        df = pd.DataFrame({
            "col": [1, None, 3, 4],  # 25% missing
        })
        result = drop_high_missing_columns(df, threshold=0.5)
        assert "col" in result.columns

    def test_no_columns_dropped_when_none_exceed_threshold(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = drop_high_missing_columns(df, threshold=0.5)
        assert result.shape == df.shape


# ── drop_duplicates ───────────────────────────────────────────────────────────

class TestDropDuplicates:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({"id": [1, 1, 2], "val": ["a", "a", "b"]})
        result = drop_duplicates(df)
        assert len(result) == 2

    def test_keeps_first_occurrence(self):
        df = pd.DataFrame({"id": [1, 1], "val": ["first", "second"]})
        result = drop_duplicates(df)
        assert result.iloc[0]["val"] == "first"

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        result = drop_duplicates(df)
        assert len(result) == len(df)


# ── clip_outliers ─────────────────────────────────────────────────────────────

class TestClipOutliers:
    def test_clips_extreme_values(self):
        data = [10] * 98 + [1000, -1000]  # two obvious outliers
        df = pd.DataFrame({"val": data})
        result = clip_outliers(df, columns=["val"], n_std=3)
        assert result["val"].max() < 1000
        assert result["val"].min() > -1000

    def test_preserves_normal_values(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        result = clip_outliers(df, columns=["val"], n_std=3)
        assert list(result["val"]) == [1, 2, 3, 4, 5]

    def test_zero_std_column_unchanged(self):
        df = pd.DataFrame({"val": [5, 5, 5, 5]})
        result = clip_outliers(df, columns=["val"], n_std=3)
        assert list(result["val"]) == [5, 5, 5, 5]


# ── impute_missing ────────────────────────────────────────────────────────────

class TestImputeMissing:
    def test_median_imputation(self):
        df = pd.DataFrame({"val": [1.0, 2.0, None, 4.0]})
        result = impute_missing(df, strategy="median")
        assert result["val"].isnull().sum() == 0
        assert result.loc[2, "val"] == 2.0  # median of [1, 2, 4]

    def test_mean_imputation(self):
        df = pd.DataFrame({"val": [0.0, 10.0, None]})
        result = impute_missing(df, strategy="mean")
        assert result.loc[2, "val"] == pytest.approx(5.0)

    def test_categorical_imputed_with_mode(self):
        df = pd.DataFrame({"cat": ["a", "a", "b", None]})
        result = impute_missing(df)
        assert result["cat"].isnull().sum() == 0
        assert result.loc[3, "cat"] == "a"  # mode

    def test_no_missing_unchanged(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        result = impute_missing(df)
        assert list(result["val"]) == [1.0, 2.0, 3.0]


# ── normalise_email ───────────────────────────────────────────────────────────

class TestNormaliseEmail:
    def test_lowercases_email(self):
        df = pd.DataFrame({"email": ["USER@EXAMPLE.COM"]})
        result = normalise_email(df)
        assert result.loc[0, "email"] == "user@example.com"

    def test_strips_whitespace(self):
        df = pd.DataFrame({"email": ["  user@example.com  "]})
        result = normalise_email(df)
        assert result.loc[0, "email"] == "user@example.com"

    def test_no_email_column_returns_unchanged(self):
        df = pd.DataFrame({"name": ["Alice"]})
        result = normalise_email(df)
        assert "email" not in result.columns


# ── strip_string_columns ──────────────────────────────────────────────────────

class TestStripStringColumns:
    def test_strips_leading_trailing_spaces(self):
        df = pd.DataFrame({"name": ["  Alice  ", "Bob "]})
        result = strip_string_columns(df)
        assert result.loc[0, "name"] == "Alice"
        assert result.loc[1, "name"] == "Bob"

    def test_numeric_columns_unaffected(self):
        df = pd.DataFrame({"name": [" A "], "age": [30]})
        result = strip_string_columns(df)
        assert result["age"].dtype == np.int64
