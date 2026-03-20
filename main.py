"""Main entry point for the product price prediction pipeline.

Usage
-----
    python main.py                      # run full pipeline
    python main.py --skip-eda           # skip EDA plots
    python main.py --data path/to/file  # use a custom CSV file
"""

import argparse
import sys
import os

# Ensure the project root is on the import path when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import ensure_dirs, set_random_seed
from src.data_preprocessing import load_data, clean_data, validate_schema
from src.eda import run_eda
from src.feature_engineering import engineer_features
from src.train import train_models
from src.evaluate import evaluate_models


RANDOM_STATE = 42
MODEL_NAMES = ["LinearRegression", "DecisionTree", "RandomForest", "GradientBoosting"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Product Price Prediction – full ML pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the products CSV file (auto-generated if not provided).",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip the EDA step (useful for faster iteration).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    ensure_dirs()

    print("\n" + "=" * 60)
    print("  PRODUCT PRICE PREDICTION – ML PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & validate data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading and cleaning data …")
    df = load_data(args.data)
    validate_schema(df)
    df = clean_data(df)

    # ------------------------------------------------------------------
    # 2. EDA
    # ------------------------------------------------------------------
    if not args.skip_eda:
        print("\n[2/4] Running EDA …")
        run_eda(df)
    else:
        print("\n[2/4] EDA skipped.")

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print("\n[3/4] Training models …")
    results = train_models(filepath=args.data, random_state=args.seed)

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    print("\n[4/4] Evaluating models on hold-out test set …")
    summary = evaluate_models(
        model_names=MODEL_NAMES,
        filepath=args.data,
        random_state=args.seed,
    )

    # Final summary
    best = summary.sort_values("RMSE").iloc[0]
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best model : {best['Model']}")
    print(f"Test RMSE  : {best['RMSE']:.2f}")
    print(f"Test R²    : {best['R2']:.4f}")
    print(f"Test MAPE  : {best['MAPE']:.2f}%")
    print("\nSaved artefacts:")
    print("  models/   – trained model files (.pkl)")
    print("  plots/    – EDA and evaluation plots (.png)")


if __name__ == "__main__":
    main()
