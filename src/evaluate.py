"""Model evaluation for product price prediction.

Provides:
  - Per-model metrics on the hold-out test set
  - Residual analysis plots
  - Feature importance plots (for tree-based models)
  - Model comparison summary table
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error

from src.utils import rmse, mape, load_model, PLOTS_DIR
from src.data_preprocessing import (
    load_data,
    clean_data,
    validate_schema,
    split_data,
    TARGET_COLUMN,
)
from src.feature_engineering import engineer_features, get_feature_columns
from src.train import _fit_preprocessors, _apply_preprocessors

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Plot saved → {path}")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of regression metrics for a single model."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_predictions_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> None:
    """Scatter plot of predicted vs. actual prices."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="none", color="steelblue", s=20)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price (USD)")
    ax.set_ylabel("Predicted Price (USD)")
    ax.set_title(f"{model_name} – Predicted vs. Actual")
    ax.legend()
    _save(fig, f"{model_name}_pred_vs_actual.png")


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> None:
    """Residual distribution and residuals-vs-fitted scatter."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=20, color="darkorange")
    axes[0].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title(f"{model_name} – Residuals vs. Fitted")

    axes[1].hist(residuals, bins=50, edgecolor="white", color="teal")
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title(f"{model_name} – Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    _save(fig, f"{model_name}_residuals.png")


def plot_feature_importance(model, model_name: str) -> None:
    """Bar chart of feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return

    feature_names = get_feature_columns()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color="steelblue",
        edgecolor="white",
    )
    ax.invert_yaxis()
    ax.set_title(f"{model_name} – Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save(fig, f"{model_name}_feature_importance.png")


def plot_model_comparison(summary: pd.DataFrame) -> None:
    """Bar chart comparing RMSE and R² across all models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    summary_sorted = summary.sort_values("RMSE")
    axes[0].barh(summary_sorted["Model"], summary_sorted["RMSE"], color="steelblue")
    axes[0].set_title("Model Comparison – RMSE (lower is better)")
    axes[0].set_xlabel("RMSE")
    axes[0].invert_yaxis()

    summary_sorted2 = summary.sort_values("R2", ascending=False)
    axes[1].barh(summary_sorted2["Model"], summary_sorted2["R2"], color="darkorange")
    axes[1].set_title("Model Comparison – R² (higher is better)")
    axes[1].set_xlabel("R²")
    axes[1].invert_yaxis()

    plt.tight_layout()
    _save(fig, "model_comparison.png")


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------
def evaluate_models(
    model_names: list[str],
    filepath: str | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate saved models on the hold-out test set.

    Parameters
    ----------
    model_names  : names used when models were saved (e.g. 'RandomForest').
    filepath     : path to the products CSV.
    random_state : must match the value used during training.
    verbose      : print metrics table and save plots.

    Returns
    -------
    summary : DataFrame with one row per model and metric columns.
    """
    # Reconstruct test set identically to training pipeline
    df = load_data(filepath)
    validate_schema(df)
    df = clean_data(df)
    df = engineer_features(df)

    train_df, val_df, test_df = split_data(df, random_state=random_state)
    encoder, scaler = _fit_preprocessors(train_df)
    X_test = _apply_preprocessors(test_df, encoder, scaler)
    y_test = test_df[TARGET_COLUMN].values

    rows = []
    for name in model_names:
        model = load_model(name)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metrics["Model"] = name
        rows.append(metrics)

        if verbose:
            print(f"\n{name}:")
            for k, v in metrics.items():
                if k != "Model":
                    print(f"  {k}: {v:.4f}")

            plot_predictions_vs_actual(y_test, y_pred, name)
            plot_residuals(y_test, y_pred, name)
            plot_feature_importance(model, name)

    summary = pd.DataFrame(rows)[["Model", "RMSE", "MAE", "MAPE", "R2"]]

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY (test set)")
        print("=" * 60)
        print(summary.to_string(index=False))
        plot_model_comparison(summary)

    return summary
