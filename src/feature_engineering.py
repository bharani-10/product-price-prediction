"""Feature engineering for product price prediction."""

import numpy as np
import pandas as pd

from src.data_preprocessing import TARGET_COLUMN, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multiplicative interaction terms between selected numeric features.

    New columns
    -----------
    rating_x_reviews    : rating × log(1 + num_reviews)
    discount_x_weight   : discount_pct × weight_kg
    """
    df = df.copy()
    df["rating_x_reviews"] = df["rating"] * np.log1p(df["num_reviews"])
    df["discount_x_weight"] = df["discount_pct"] * df["weight_kg"]
    return df


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed versions of right-skewed features.

    New columns
    -----------
    log_num_reviews : log(1 + num_reviews)
    """
    df = df.copy()
    df["log_num_reviews"] = np.log1p(df["num_reviews"])
    return df


def add_price_tier_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary flag indicating whether the product is discounted.

    New column
    ----------
    is_discounted : 1 if discount_pct > 0, else 0
    """
    df = df.copy()
    df["is_discounted"] = (df["discount_pct"] > 0).astype(int)
    return df


def add_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a composite 'value score' combining rating and reviews.

    value_score = rating × log(1 + num_reviews) / (1 + discount_pct / 100)

    This approximates perceived customer value adjusted for discount.
    """
    df = df.copy()
    df["value_score"] = (
        df["rating"] * np.log1p(df["num_reviews"]) / (1 + df["discount_pct"] / 100)
    )
    return df


# ---------------------------------------------------------------------------
# Engineered feature registry
# ---------------------------------------------------------------------------
ENGINEERED_NUMERIC = [
    "rating_x_reviews",
    "discount_x_weight",
    "log_num_reviews",
    "is_discounted",
    "value_score",
]

ALL_NUMERIC = NUMERIC_COLUMNS + ENGINEERED_NUMERIC


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return the enriched DataFrame.

    This function is the single entry-point used by the training pipeline.
    The order of transformations is deterministic so that the same feature
    matrix is produced at inference time.
    """
    df = add_log_features(df)
    df = add_interaction_features(df)
    df = add_price_tier_flag(df)
    df = add_value_score(df)
    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names used after engineering."""
    return list(CATEGORICAL_COLUMNS) + list(ALL_NUMERIC)
