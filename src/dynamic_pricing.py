
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pickle
def estimate_elasticity(data, feature_list, target_variable):
    """
    Estimate demand or supply elasticity using OLS regression.

    Parameters:
        data (pd.DataFrame): The input dataset.
        feature_list (list): List of features to be used for the model.
        target_variable (str): The target variable for the model.
        elasticity_type (str): Type of elasticity to estimate ('demand' or 'supply').

    Returns:
        tuple: Estimated elasticity and model summary.
    """
    
   
    X = data[feature_list]
    y = data[target_variable]

    # Split the dataset into training and test sets first to avoid data leakage.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify which features are categorical vs. numeric in the training data
    categorical_features = [feature for feature in feature_list 
                            if X_train[feature].dtype == 'object' or pd.api.types.is_categorical_dtype(X_train[feature])]
    numeric_features = [feature for feature in feature_list if feature not in categorical_features]

    # Define a transformer to log-transform numeric features (adding 1 to avoid log(0))
    log_transformer = FunctionTransformer(lambda x: np.log(x + 1))
    
    # Build a preprocessor pipeline for the training data:
    # - Log-transforms then scales numeric features.
    # - One-hot encodes categorical features.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('log', log_transformer),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    
    # Fit the preprocessor on the training data and transform the training set
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Log-transform the target variable for the training data
    y_train_log = np.log(y_train + 1)
    
    # Add a constant term for the intercept to the training design matrix
    X_train_design = sm.add_constant(X_train_transformed)

    # Fit the OLS regression model on the log–log data from the training set
    model = sm.OLS(y_train_log, X_train_design).fit()
    print(model.summary())
    
    # Extract the elasticity coefficient for 'Historical_Cost_of_Ride'
    if 'Historical_Cost_of_Ride' in numeric_features:
        # Because we added a constant at the beginning, the index is shifted by 1
        coef_index = numeric_features.index('Historical_Cost_of_Ride') + 1
        elasticity = model.params[coef_index]
    else:
        elasticity = None
        print("Error: 'Historical_Cost_of_Ride' is not among the numeric features.")
    
    return elasticity, model.summary()

# Example usage for demand elasticity estimation
def estimate_demand_elasticity(data):
    demand_features = [
        'Number_of_Drivers', 'Location_Category', 'Customer_Loyalty_Status',
        'Number_of_Past_Rides', 'Average_Ratings', 'Time_of_Booking',
        'Vehicle_Type', 'Expected_Ride_Duration', 'Historical_Cost_of_Ride'
    ]
    demand_target = 'Number_of_Riders'
    estimated_demand_elasticity, demand_summary = estimate_elasticity(data, demand_features, demand_target)
    # print(f"Estimated Demand Elasticity: {estimated_demand_elasticity}")
    return estimated_demand_elasticity, demand_summary

# Example usage for supply elasticity estimation
def estimate_supply_elasticity(data):
    supply_features = [
        'Number_of_Riders', 'Location_Category', 'Customer_Loyalty_Status',
        'Number_of_Past_Rides', 'Average_Ratings', 'Time_of_Booking',
        'Vehicle_Type', 'Expected_Ride_Duration', 'Historical_Cost_of_Ride'
    ]
    supply_target = 'Number_of_Drivers'
    estimated_supply_elasticity, supply_summary = estimate_elasticity(data, supply_features, supply_target)
    # print(f"Estimated Supply Elasticity: {estimated_supply_elasticity}")
    return estimated_supply_elasticity, supply_summary
