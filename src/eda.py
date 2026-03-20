"""Exploratory Data Analysis for the product price dataset."""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import PLOTS_DIR
from src.data_preprocessing import (
    TARGET_COLUMN,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")


def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Plot saved → {path}")


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of the price column."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df[TARGET_COLUMN], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Price distribution")
    axes[0].set_xlabel("Price (USD)")
    axes[0].set_ylabel("Count")

    log_prices = np.log1p(df[TARGET_COLUMN])
    axes[1].hist(log_prices, bins=50, color="darkorange", edgecolor="white")
    axes[1].set_title("Log(1 + Price) distribution")
    axes[1].set_xlabel("log(1 + Price)")

    fig.suptitle("Target Variable – Price", fontsize=14)
    _save(fig, "target_distribution.png")


def plot_categorical_vs_price(df: pd.DataFrame) -> None:
    """Box plots of price for each categorical feature."""
    fig, axes = plt.subplots(1, len(CATEGORICAL_COLUMNS), figsize=(14, 5))
    if len(CATEGORICAL_COLUMNS) == 1:
        axes = [axes]

    for ax, col in zip(axes, CATEGORICAL_COLUMNS):
        order = df.groupby(col)[TARGET_COLUMN].median().sort_values().index.tolist()
        sns.boxplot(data=df, x=col, y=TARGET_COLUMN, order=order, ax=ax)
        ax.set_title(f"Price by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Price (USD)")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Categorical Features vs. Price", fontsize=14)
    plt.tight_layout()
    _save(fig, "categorical_vs_price.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Histograms for each numeric feature."""
    cols = NUMERIC_COLUMNS
    n_cols = 3
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col], bins=40, edgecolor="white", color="teal")
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions", fontsize=14)
    plt.tight_layout()
    _save(fig, "numeric_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap for numeric columns (including target)."""
    numeric_df = df[NUMERIC_COLUMNS + [TARGET_COLUMN]].copy()
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap", fontsize=14)
    _save(fig, "correlation_heatmap.png")


def plot_scatter_matrix(df: pd.DataFrame) -> None:
    """Pair plot of a subset of features vs. price."""
    cols = ["rating", "num_reviews", "discount_pct", "weight_kg", TARGET_COLUMN]
    sample = df[cols].sample(min(500, len(df)), random_state=42)
    fig = sns.pairplot(sample, y_vars=[TARGET_COLUMN], x_vars=cols[:-1], height=3)
    fig.figure.suptitle("Scatter Plot: Features vs. Price", y=1.02, fontsize=14)
    _save(fig.figure, "scatter_matrix.png")


def plot_price_by_category_and_brand(df: pd.DataFrame) -> None:
    """Grouped bar chart of median price by category and brand tier."""
    pivot = (
        df.groupby(["category", "brand_tier"])[TARGET_COLUMN]
        .median()
        .reset_index()
        .pivot(index="category", columns="brand_tier", values=TARGET_COLUMN)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title("Median Price by Category and Brand Tier", fontsize=14)
    ax.set_xlabel("Category")
    ax.set_ylabel("Median Price (USD)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Brand Tier")
    plt.tight_layout()
    _save(fig, "price_by_category_brand.png")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print descriptive statistics for all columns."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(df.describe(include="all").T.to_string())

    print("\nValue counts for categorical features:")
    for col in CATEGORICAL_COLUMNS:
        print(f"\n  {col}:\n{df[col].value_counts()}")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    """Run the full EDA pipeline and save all plots."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print_summary_statistics(df)
    plot_target_distribution(df)
    plot_categorical_vs_price(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_scatter_matrix(df)
    plot_price_by_category_and_brand(df)
    print("\nEDA complete. All plots saved to the 'plots/' directory.")
