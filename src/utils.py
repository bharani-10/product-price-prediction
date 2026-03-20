"""Utility functions for the product price prediction project."""

import os
import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")


def ensure_dirs() -> None:
    """Create required project directories if they do not already exist."""
    for directory in (DATA_DIR, MODELS_DIR, PLOTS_DIR):
        os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------
def save_model(model, name: str) -> str:
    """Persist *model* to ``models/<name>.pkl`` and return the full path."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Model saved → {path}")
    return path


def load_model(name: str):
    """Load a previously saved model from ``models/<name>.pkl``."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the Mean Absolute Percentage Error (%)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    """Set the NumPy random seed for reproducibility."""
    np.random.seed(seed)


def display_df_info(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Print basic shape and head information for a DataFrame."""
    print(f"\n{'=' * 60}")
    print(f"{label}  — shape: {df.shape}")
    print("=" * 60)
    print(df.head())
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
