import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a model and return metrics"""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 {model_name} Results:")
    print(f"   MAE  = {mae:.2f}")
    print(f"   MSE  = {mse:.2f}")
    print(f"   RMSE = {rmse:.2f}")
    print(f"   R²   = {r2:.4f}")
    
    return {
        'model_name': model_name,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
