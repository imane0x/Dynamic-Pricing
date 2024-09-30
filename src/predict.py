# src/predict.py

from joblib import load
from config import MODEL_PATH

def predict(X):
    """
    Predicts the target variable using the trained model.

    Args:
        X: Input features for prediction.

    Returns:
        Predicted values.
    """
    model = load(MODEL_PATH)
    return model.predict(X)
