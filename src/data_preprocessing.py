"""Data loading and preprocessing for product price prediction."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from src.utils import DATA_DIR, display_df_info


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COLUMN = "price"
CATEGORICAL_COLUMNS = ["category", "brand_tier"]
NUMERIC_COLUMNS = [
    "rating",
    "num_reviews",
    "discount_pct",
    "weight_kg",
    "warranty_years",
    "is_new_arrival",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(filepath: str | None = None) -> pd.DataFrame:
    """Load the products CSV dataset.

    If *filepath* is not provided the function looks for ``products.csv``
    inside the project ``data/`` directory.  When the file does not exist it
    auto-generates a synthetic dataset and saves it first.
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "products.csv")

    if not os.path.exists(filepath):
        print("Dataset not found – generating synthetic data …")
        from data.generate_data import generate_dataset  # local import to avoid circular dep
        df = generate_dataset()
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")
    else:
        df = pd.read_csv(filepath)

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if required columns are missing."""
    required = set(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of *df*.

    Steps
    -----
    1. Drop duplicate rows.
    2. Drop rows where the target is missing or non-positive.
    3. Clip numeric outliers to ±3 standard deviations.
    4. Fill remaining numeric NaNs with column median.
    5. Fill categorical NaNs with the mode.
    """
    df = df.copy()

    # 1. Remove exact duplicates
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = n_before - len(df)
    if dropped:
        print(f"Removed {dropped} duplicate row(s).")

    # 2. Drop invalid target rows
    df = df[df[TARGET_COLUMN].notna() & (df[TARGET_COLUMN] > 0)]

    # 3. Clip numeric outliers
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)

    # 4. Impute numeric NaNs
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # 5. Impute categorical NaNs
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) splits."""
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=random_state
    )
    print(
        f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Encoding & Scaling
# ---------------------------------------------------------------------------
def encode_and_scale(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ordinal-encode categoricals, standard-scale numerics.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test  (as numpy arrays)
    """
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    scaler = StandardScaler()

    feature_cols = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

    # Fit on train only
    train_cat = encoder.fit_transform(train[CATEGORICAL_COLUMNS])
    train_num = scaler.fit_transform(train[NUMERIC_COLUMNS])
    X_train = np.hstack([train_cat, train_num])
    y_train = train[TARGET_COLUMN].values

    # Transform val and test
    val_cat = encoder.transform(val[CATEGORICAL_COLUMNS])
    val_num = scaler.transform(val[NUMERIC_COLUMNS])
    X_val = np.hstack([val_cat, val_num])
    y_val = val[TARGET_COLUMN].values

    test_cat = encoder.transform(test[CATEGORICAL_COLUMNS])
    test_num = scaler.transform(test[NUMERIC_COLUMNS])
    X_test = np.hstack([test_cat, test_num])
    y_test = test[TARGET_COLUMN].values

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def load_and_preprocess(
    filepath: str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    verbose: bool = True,
):
    """End-to-end data loading → cleaning → splitting → encoding.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = load_data(filepath)
    if verbose:
        display_df_info(df, "Raw dataset")

    validate_schema(df)
    df = clean_data(df)
    if verbose:
        display_df_info(df, "Cleaned dataset")

    train, val, test = split_data(df, test_size, val_size, random_state)
    return encode_and_scale(train, val, test)
