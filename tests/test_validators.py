"""
test_validators.py
------------------
Tests for DataValidator. Focuses on boundary conditions —
the edge cases that bite you in production at 2am.
"""

import pandas as pd
import pytest

from src.validators import DataValidator


@pytest.fixture
def valid_df():
    """A minimal valid customer DataFrame."""
    return pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "email": ["a@example.com", "b@example.com", "c@example.com"],
        "signup_date": ["2023-01-01", "2023-06-15", "2024-03-10"],
        "country": ["US", "UK", "DE"],
        "age": [25, 34, 45],
        "revenue": [100.0, 250.0, 75.0],
    })


class TestDataValidator:
    def setup_method(self):
        self.v = DataValidator()

    def test_valid_dataframe_passes(self, valid_df):
        report = self.v.validate(valid_df)
        assert report.passed

    def test_empty_dataframe_fails(self):
        report = self.v.validate(pd.DataFrame())
        assert not report.passed
        assert any("empty" in e.lower() for e in report.errors)

    def test_missing_required_column_fails(self, valid_df):
        df = valid_df.drop(columns=["email"])
        report = self.v.validate(df)
        assert not report.passed
        assert any("email" in e for e in report.errors)

    def test_duplicate_rows_produce_warning_not_error(self, valid_df):
        df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
        report = self.v.validate(df)
        assert report.passed                       # warning, not error
        assert any("duplicate" in w.lower() for w in report.warnings)

    def test_high_missing_rate_produces_warning(self, valid_df):
        valid_df["sparse_col"] = [None, None, None]  # 100% null
        report = self.v.validate(valid_df)
        assert report.passed
        assert any("sparse_col" in w for w in report.warnings)

    def test_invalid_email_produces_warning(self, valid_df):
        valid_df.loc[0, "email"] = "not-an-email"
        report = self.v.validate(valid_df)
        assert report.passed
        assert any("invalid email" in w.lower() for w in report.warnings)

    def test_numeric_outliers_produce_warnings(self, valid_df):
        # Need many similar values + one extreme to exceed 3σ threshold
        extra_rows = pd.DataFrame({
            "customer_id": [f"C{i:03d}" for i in range(10, 30)],
            "email": [f"u{i}@x.com" for i in range(10, 30)],
            "signup_date": ["2023-01-01"] * 20,
            "country": ["US"] * 20,
            "age": [30] * 20,
            "revenue": [100.0] * 20,
        })
        df = pd.concat([valid_df, extra_rows], ignore_index=True)
        df.loc[0, "revenue"] = 999_999   # clear outlier against a tight cluster
        report = self.v.validate(df)
        assert any("outlier" in w.lower() for w in report.warnings)

    def test_multiple_missing_required_columns(self, valid_df):
        df = valid_df.drop(columns=["email", "country"])
        report = self.v.validate(df)
        assert not report.passed
        # Both missing columns should be mentioned
        combined_errors = " ".join(report.errors)
        assert "email" in combined_errors
        assert "country" in combined_errors
