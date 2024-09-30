# src/config.py

# Paths
DATA_PATH = "data/dynamic_pricing.csv"          # Path to the dataset
MODEL_PATH = "models/best_model.pkl"  # Path to save/load the trained model
PIPELINE_PATH = "models/pipeline.pkl"
# Data split parameters
TEST_SIZE = 0.3                      # Proportion of the dataset to include in the test split
RANDOM_STATE = 42                    # Seed for reproducibility

# Grid Search parameters for Random Forest
GRID_SEARCH_PARAMS = {
    'n_estimators': [100, 200, 300] , # Number of trees in the forest
     'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10]   # Minimum number of samples required to split an internal node
}
