"""Model training for product price prediction.

Trains four regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor

Each model is evaluated on the validation set during training and the best
model (lowest RMSE) is saved to disk.
"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import pandas as pd

from src.utils import rmse, mape, save_model, DATA_DIR, MODELS_DIR
from src.data_preprocessing import (
    load_data,
    clean_data,
    validate_schema,
    split_data,
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
)
from src.feature_engineering import engineer_features, get_feature_columns, ALL_NUMERIC


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def get_models(random_state: int = 42) -> dict:
    """Return a dict mapping model name → unfitted estimator."""
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=10,
            random_state=random_state,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# Preprocessing helpers (fit-on-train, transform-all)
# ---------------------------------------------------------------------------
def _fit_preprocessors(train_df: pd.DataFrame):
    """Fit and return (encoder, scaler) on the training split."""
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    scaler = StandardScaler()

    encoder.fit(train_df[CATEGORICAL_COLUMNS])
    scaler.fit(train_df[ALL_NUMERIC])
    return encoder, scaler


def _apply_preprocessors(df: pd.DataFrame, encoder, scaler) -> np.ndarray:
    """Transform *df* using pre-fitted encoder and scaler."""
    cat = encoder.transform(df[CATEGORICAL_COLUMNS])
    num = scaler.transform(df[ALL_NUMERIC])
    return np.hstack([cat, num])


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def train_models(
    filepath: str | None = None,
    random_state: int = 42,
    cv_folds: int = 5,
    verbose: bool = True,
) -> dict:
    """Full training pipeline.

    Parameters
    ----------
    filepath      : path to products CSV; auto-generates data if None.
    random_state  : seed for reproducibility.
    cv_folds      : number of cross-validation folds.
    verbose       : whether to print progress.

    Returns
    -------
    results : dict with keys = model names and values = dicts containing
              'model', 'val_rmse', 'val_mape', 'cv_rmse_mean', 'cv_rmse_std'.
    """
    # 1. Load & clean
    df = load_data(filepath)
    validate_schema(df)
    df = clean_data(df)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Split
    train_df, val_df, test_df = split_data(df, random_state=random_state)

    # 4. Encode / scale
    encoder, scaler = _fit_preprocessors(train_df)
    X_train = _apply_preprocessors(train_df, encoder, scaler)
    X_val = _apply_preprocessors(val_df, encoder, scaler)
    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values

    # Persist preprocessors so they can be reused at inference time
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_model(encoder, "encoder")
    save_model(scaler, "scaler")

    models = get_models(random_state)
    results = {}

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

    for name, model in models.items():
        if verbose:
            print(f"\nTraining {name} …")

        model.fit(X_train, y_train)

        # Validation metrics
        val_pred = model.predict(X_val)
        val_rmse = rmse(y_val, val_pred)
        val_mape_val = mape(y_val, val_pred)

        # Cross-validation on training data
        cv_scores = cross_val_score(
            model, X_train, y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv_folds,
        )
        cv_rmse_mean = float(-cv_scores.mean())
        cv_rmse_std = float(cv_scores.std())

        results[name] = {
            "model": model,
            "val_rmse": val_rmse,
            "val_mape": val_mape_val,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
        }

        if verbose:
            print(
                f"  Val RMSE: {val_rmse:.2f} | Val MAPE: {val_mape_val:.2f}% | "
                f"CV RMSE: {cv_rmse_mean:.2f} ± {cv_rmse_std:.2f}"
            )

        # Save each trained model
        save_model(model, name)

    # Identify best model by validation RMSE
    best_name = min(results, key=lambda k: results[k]["val_rmse"])
    if verbose:
        print(f"\nBest model: {best_name}  (Val RMSE = {results[best_name]['val_rmse']:.2f})")
    save_model(results[best_name]["model"], "best_model")

    return results
