"""Generate a synthetic product-price dataset and save it to data/products.csv."""

import os
import numpy as np
import pandas as pd


def generate_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Return a synthetic product DataFrame with realistic price relationships.

    Features
    --------
    category        : product category (Electronics, Clothing, Home, Sports, Books)
    brand_tier      : brand quality tier (Budget, Mid, Premium)
    rating          : average customer rating (1.0 – 5.0)
    num_reviews     : number of customer reviews
    discount_pct    : discount percentage applied (0 – 50 %)
    weight_kg       : product weight in kilograms
    warranty_years  : warranty duration in years
    is_new_arrival  : 1 if listed as new arrival, else 0

    Target
    ------
    price           : product price in USD
    """
    rng = np.random.default_rng(random_state)

    categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
    brand_tiers = ["Budget", "Mid", "Premium"]

    # --- categorical features -----------------------------------------------
    category = rng.choice(categories, size=n_samples)
    brand_tier = rng.choice(brand_tiers, size=n_samples)

    # --- numeric features ---------------------------------------------------
    rating = rng.uniform(1.0, 5.0, size=n_samples).round(1)
    num_reviews = rng.integers(10, 5000, size=n_samples)
    discount_pct = rng.uniform(0, 50, size=n_samples).round(1)
    weight_kg = rng.uniform(0.1, 20.0, size=n_samples).round(2)
    warranty_years = rng.choice([0, 1, 2, 3, 5], size=n_samples)
    is_new_arrival = rng.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # --- price construction (deterministic signal + noise) ------------------
    category_base = {
        "Electronics": 300,
        "Clothing": 60,
        "Home": 120,
        "Sports": 80,
        "Books": 20,
    }
    tier_multiplier = {"Budget": 0.6, "Mid": 1.0, "Premium": 2.5}

    base = np.array([category_base[c] for c in category], dtype=float)
    multiplier = np.array([tier_multiplier[t] for t in brand_tier], dtype=float)

    price = (
        base * multiplier
        + rating * 15
        + np.log1p(num_reviews) * 5
        - discount_pct * 0.5
        + weight_kg * 3
        + warranty_years * 10
        + is_new_arrival * 20
        + rng.normal(0, 20, size=n_samples)
    ).round(2)

    # ensure no negative prices
    price = np.clip(price, 1.0, None)

    df = pd.DataFrame(
        {
            "category": category,
            "brand_tier": brand_tier,
            "rating": rating,
            "num_reviews": num_reviews,
            "discount_pct": discount_pct,
            "weight_kg": weight_kg,
            "warranty_years": warranty_years,
            "is_new_arrival": is_new_arrival,
            "price": price,
        }
    )
    return df


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "products.csv")
    df = generate_dataset()
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}  ({len(df)} rows)")
    print(df.head())
