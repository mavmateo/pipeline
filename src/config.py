"""
config.py
---------
Loads and validates pipeline configuration using Pydantic.
Single source of truth — all other modules import from here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, field_validator


# ── Nested config models ──────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    name: str
    version: str
    environment: str


class DataConfig(BaseModel):
    raw_path: str
    processed_path: str
    supported_formats: List[str]
    encoding: str
    chunk_size: int


class CleaningConfig(BaseModel):
    drop_duplicate_subset: Optional[List[str]]
    missing_value_threshold: float
    outlier_std_threshold: float
    date_formats: List[str]

    @field_validator("missing_value_threshold", "outlier_std_threshold")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Threshold must be positive")
        return v


class SchemaConfig(BaseModel):
    required_columns: List[str]
    column_types: Dict[str, str]


class LoggingConfig(BaseModel):
    level: str
    log_to_file: bool
    log_path: str
    max_bytes: int
    backup_count: int


class AppConfig(BaseModel):
    pipeline: PipelineConfig
    data: DataConfig
    cleaning: CleaningConfig
    schema_: SchemaConfig
    logging: LoggingConfig

    class Config:
        populate_by_name = True


# ── Loader ────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.getenv(
    "PIPELINE_CONFIG",
    str(Path(__file__).parent.parent / "configs" / "settings.yaml"),
)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load config once and cache. Override path via PIPELINE_CONFIG env var."""
    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    # 'schema' is a reserved name in Pydantic v2 — rename key
    raw["schema_"] = raw.pop("schema")
    return AppConfig(**raw)
