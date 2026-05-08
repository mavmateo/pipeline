"""
medallion.py
------------
Defines the pipeline layers used by the medallion architecture.

Bronze keeps data close to the source with light standardisation.
Silver produces a clean, conformed dataset.
Gold adds business-ready transformations for analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple

import pandas as pd

from src import cleaners, transformers

DataStep = Callable[[pd.DataFrame], pd.DataFrame]


class MedallionLevel(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"

    @classmethod
    def choices(cls) -> list[str]:
        return [level.value for level in cls]

    @classmethod
    def from_value(cls, value: str | "MedallionLevel") -> "MedallionLevel":
        if isinstance(value, cls):
            return value
        normalised = value.strip().lower()
        try:
            return cls(normalised)
        except ValueError as exc:
            choices = ", ".join(cls.choices())
            raise ValueError(
                f"Unsupported medallion level '{value}'. Expected one of: {choices}"
            ) from exc


@dataclass(frozen=True)
class MedallionLayer:
    level: MedallionLevel
    description: str
    steps: Tuple[DataStep, ...]


LAYER_DEFINITIONS: dict[MedallionLevel, MedallionLayer] = {
    MedallionLevel.BRONZE: MedallionLayer(
        level=MedallionLevel.BRONZE,
        description="Raw landing layer with light source standardisation",
        steps=(
            cleaners.normalize_column_names,
            cleaners.strip_string_columns,
        ),
    ),
    MedallionLevel.SILVER: MedallionLayer(
        level=MedallionLevel.SILVER,
        description="Clean, deduplicated, typed, privacy-safe data",
        steps=(
            cleaners.drop_empty_columns,
            cleaners.drop_high_missing_columns,
            cleaners.drop_duplicates,
            cleaners.normalise_email,
            cleaners.remove_pii_columns,
            cleaners.cast_column_types,
            cleaners.clip_outliers,
            cleaners.impute_missing,
        ),
    ),
    MedallionLevel.GOLD: MedallionLayer(
        level=MedallionLevel.GOLD,
        description="Business-ready analytics transformations",
        steps=(
            transformers.normalise_country_code,
            transformers.add_customer_tenure_days,
            transformers.add_age_bucket,
            transformers.flag_high_value_customers,
        ),
    ),
}

LAYER_ORDER: tuple[MedallionLevel, ...] = (
    MedallionLevel.BRONZE,
    MedallionLevel.SILVER,
    MedallionLevel.GOLD,
)


def get_medallion_layers(
    target_level: str | MedallionLevel,
) -> tuple[MedallionLayer, ...]:
    """
    Return the cumulative layers required to produce the requested level.

    Requesting silver runs bronze + silver. Requesting gold runs all layers.
    """
    level = MedallionLevel.from_value(target_level)
    end_index = LAYER_ORDER.index(level) + 1
    return tuple(LAYER_DEFINITIONS[layer] for layer in LAYER_ORDER[:end_index])
