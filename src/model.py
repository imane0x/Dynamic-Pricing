from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump
from config import GRID_SEARCH_PARAMS, MODEL_PATH
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def train_model(X_train, y_train):
    """
    Trains the Random Forest model using Grid Search to find the best parameters.

    Args:
        X_train: Features of the training set.
        y_train: Target variable of the training set.

    Returns:
        The best trained Random Forest model.
    """
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, GRID_SEARCH_PARAMS, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    dump(grid_search.best_estimator_, MODEL_PATH)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.

    Parameters:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target variable.

    Returns:
        dict: Evaluation metrics including MAE, MSE, and R2 score.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred)
    }
    return metrics
