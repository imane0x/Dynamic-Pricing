# src/elasticity_estimation.py

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
    
    # Split data into predictors (X) and target (y)
    X = data[feature_list]
    y = data[target_variable]

    # Identify categorical features for preprocessing
    categorical_features = [feature for feature in feature_list if data[feature].dtype == 'object' or pd.api.types.is_categorical_dtype(data[feature])]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ],
        remainder='passthrough'  # Keep other numerical features as they are
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())  # Optional scaling
    ])
   

    # Transform the features
    X_processed = pipeline.fit_transform(X)
    X_processed = sm.add_constant(X_processed)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Fit the OLS regression model
    model = sm.OLS(y_train, X_train).fit()
    
    # Print the model summary
    print(model.summary())

    # Estimating elasticity based on the 'Historical_Cost_of_Ride' index
    # Adjust the index if the column order changes after preprocessing
    elasticity_index = list(X.columns).index('Historical_Cost_of_Ride')
    elasticity = model.params[elasticity_index]
    
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
