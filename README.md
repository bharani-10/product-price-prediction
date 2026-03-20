# Product Price Prediction

End-to-end machine learning project for predicting product prices using regression models.  
Includes data preprocessing, exploratory data analysis (EDA), feature engineering, and a comparison of four regression algorithms.

## Project Structure

```
product-price-prediction/
├── data/
│   └── generate_data.py        # Synthetic dataset generator
├── src/
│   ├── __init__.py
│   ├── utils.py                # Shared helpers (paths, metrics, model I/O)
│   ├── data_preprocessing.py  # Loading, cleaning, splitting, encoding
│   ├── eda.py                  # Exploratory data analysis & plots
│   ├── feature_engineering.py # Feature engineering pipeline
│   ├── train.py               # Model training (4 regressors + CV)
│   └── evaluate.py            # Test-set evaluation & comparison plots
├── models/                    # Saved model artefacts (auto-created)
├── plots/                     # EDA & evaluation plots  (auto-created)
├── main.py                    # Pipeline entry point
└── requirements.txt
```

## Models

| Model | Notes |
|---|---|
| Linear Regression | Baseline model |
| Decision Tree | Depth-limited tree regressor |
| Random Forest | 200-tree ensemble |
| Gradient Boosting | 200-stage boosting with subsampling |

## Features

**Raw**
- `category` – product category (Electronics, Clothing, Home, Sports, Books)
- `brand_tier` – Budget / Mid / Premium
- `rating` – average customer rating (1–5)
- `num_reviews` – number of customer reviews
- `discount_pct` – discount percentage (0–50 %)
- `weight_kg` – product weight in kg
- `warranty_years` – warranty duration in years
- `is_new_arrival` – binary new-arrival flag

**Engineered**
- `log_num_reviews` – log(1 + num_reviews)
- `rating_x_reviews` – rating × log(1 + num_reviews)
- `discount_x_weight` – discount_pct × weight_kg
- `is_discounted` – binary discount flag
- `value_score` – composite rating/review/discount score

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (auto-generates data if needed)
python main.py

# 3. Skip EDA for a faster run
python main.py --skip-eda

# 4. Use your own CSV file
python main.py --data path/to/your/products.csv
```

### Expected CSV columns

`category`, `brand_tier`, `rating`, `num_reviews`, `discount_pct`,  
`weight_kg`, `warranty_years`, `is_new_arrival`, `price`

## Output

- **`models/`** – serialised model files (`.pkl`) for each algorithm plus `best_model.pkl`
- **`plots/`** – EDA plots (distributions, correlations, category breakdowns) and evaluation plots (predicted vs. actual, residuals, feature importances, model comparison)

## Results (synthetic data, default seed)

| Model | Test RMSE | Test R² | Test MAPE |
|---|---|---|---|
| Linear Regression | ~153 | ~0.27 | ~41 % |
| Decision Tree | ~31 | ~0.97 | ~11 % |
| Random Forest | ~26 | ~0.98 | ~9 % |
| **Gradient Boosting** | **~23** | **~0.98** | **~8 %** |
