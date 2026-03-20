import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.preprocess import load_and_clean, encode_and_scale
from src.evaluate import evaluate_model


def train_all_models():
    """Main training pipeline"""
    print("=" * 60)
    print("🚀 PRODUCT PRICE PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and clean data
    print("\n📁 Step 1: Loading dataset...")
    df = load_and_clean('data/products.csv')
    
    # Step 2: EDA summary
    print("\n📊 Step 2: Quick EDA...")
    print(df.describe())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Step 3: Encode and scale
    print("\n⚙️ Step 3: Preprocessing...")
    X, y, scaler, encoders, feature_names = encode_and_scale(df, target_col='retail_price')
    
    # Step 4: Train-test split
    print("\n✂️ Step 4: Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    # Step 5: Train models
    print("\n🤖 Step 5: Training models...")
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        trained_models[name] = model
    
    # Step 6: Compare and pick best
    print("\n" + "=" * 60)
    print("🏆 MODEL COMPARISON:")
    print("=" * 60)
    results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
    print(results_df.to_string(index=False))
    
    best_name = results_df.iloc[0]['model_name']
    best_model = trained_models[best_name]
    best_r2 = results_df.iloc[0]['r2']
    
    print(f"\n✅ Best Model: {best_name} (R² = {best_r2:.4f})")
    
    # Step 7: Save model
    os.makedirs('models', exist_ok=True)
    artifact = {
        'model': best_model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': feature_names,
        'model_name': best_name
    }
    joblib.dump(artifact, 'models/best_model.pkl')
    print(f"\n💾 Model saved to models/best_model.pkl")
    print("=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    train_all_models()
