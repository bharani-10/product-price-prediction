import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean(filepath):
    """Load dataset and perform cleaning"""
    df = pd.read_csv(filepath)

    # Keep relevant columns (adjust based on your CSV columns)
    # Common columns: product_name, retail_price, discounted_price, 
    # discount, rating, overall_rating, brand, product_category_tree
    
    # --- Adjust these column names to match YOUR csv ---
    cols_needed = [
        'retail_price', 'discounted_price', 'discount',
        'rating', 'overall_rating', 'brand', 'product_category_tree'
    ]
    # Keep only columns that exist
    cols_needed = [c for c in cols_needed if c in df.columns]
    df = df[cols_needed].copy()

    # Clean price columns - remove currency symbols
    for col in ['retail_price', 'discounted_price']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            )

    # Clean discount column
    if 'discount' in df.columns:
        df['discount'] = pd.to_numeric(
            df['discount'].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        )

    # Extract main category from category tree
    if 'product_category_tree' in df.columns:
        df['main_category'] = df['product_category_tree'].astype(str).apply(
            lambda x: x.split('>>')[0].strip(' []"') if '>>' in str(x) else 'Other'
        )
        df.drop('product_category_tree', axis=1, inplace=True)

    # Clean rating
    if 'overall_rating' in df.columns:
        df['overall_rating'] = pd.to_numeric(df['overall_rating'], errors='coerce')

    # Drop rows where target (retail_price) is missing
    if 'retail_price' in df.columns:
        df.dropna(subset=['retail_price'], inplace=True)

    # Fill missing numeric values with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill missing categorical values
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna('Unknown', inplace=True)

    print(f"✅ Cleaned data shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def encode_and_scale(df, target_col='retail_price'):
    """Encode categorical variables and scale features"""
    encoders = {}
    
    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        # Keep only top 50 categories to avoid too many classes
        top = df[col].value_counts().nlargest(50).index
        df[col] = df[col].where(df[col].isin(top), other='Other')
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    print(f"✅ Features: {list(X.columns)}")
    print(f"   Encoded {len(cat_cols)} categorical columns: {cat_cols}")
    return X_scaled, y, scaler, encoders, X.columns.tolist()
