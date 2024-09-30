import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import load
def calculate_dynamic_price(data, demand_elasticity, supply_elasticity):
    """
    Calculate dynamic prices based on demand and supply conditions.

    Parameters:
        data (pd.DataFrame): The input dataset with relevant features.
        demand_elasticity (float): Estimated demand elasticity value.
        supply_elasticity (float): Estimated supply elasticity value.

    Returns:
        pd.DataFrame: The dataset with an additional 'dynamic_price' column.
    """

    # Define time of day adjustments based on the new categories
    time_of_day_adjustments = {
        'Morning': 1.0,    # No change
        'Afternoon': 1.1,  # Increase price by 10%
        'Evening': 1.2,    # Increase price by 20%
        'Night': 1.3       # Increase price by 30%
    }
    # Apply the adjustment based on the Time_of_Booking category
    data['time_of_day_multiplier'] = data['Time_of_Booking'].map(time_of_day_adjustments)

    # Define loyalty adjustment (e.g., loyal customers get a discount)
    loyalty_adjustments = {
        'Gold': 0.9,     # 10% discount
        'Silver': 0.98,  # 5% discount
        'Regular': 1.0   # No discount
    }
    data['loyalty_discount'] = data['Customer_Loyalty_Status'].map(loyalty_adjustments)

    # Define vehicle type adjustment (e.g., premium vehicles have a higher base price)
    vehicle_type_adjustments = {
        'Economy': 1.0,    # No change for economy
        'Premium': 1.5     # 50% higher for premium vehicles
    }
    data['vehicle_type_multiplier'] = data['Vehicle_Type'].map(vehicle_type_adjustments)

    # Define location category adjustment (e.g., urban areas may have higher prices)
    location_category_adjustments = {
        'Urban': 1.2,      # 20% increase for urban areas
        'Suburban': 1.1,   # 10% increase for suburban areas
        'Rural': 0.9       # 10% decrease for rural areas
    }
    data['location_category_multiplier'] = data['Location_Category'].map(location_category_adjustments)

    # Price adjustment factors based on elasticity and additional factors
    demand_multiplier = np.where(
        data['Number_of_Riders'] > np.percentile(data['Number_of_Riders'], 75),
        1 + demand_elasticity,  # Increase price if high demand
        1 - demand_elasticity   # Decrease price if low demand
    )

    supply_multiplier = np.where(
        data['Number_of_Drivers'] < np.percentile(data['Number_of_Drivers'], 25),
        1 + supply_elasticity,  # Increase price if low supply
        1 - supply_elasticity   # Decrease price if high supply
    )

    # Calculate dynamic prices
    data['dynamic_price'] = (
        data['Historical_Cost_of_Ride']
        * demand_multiplier
        * supply_multiplier
        * data['loyalty_discount']
        * data['time_of_day_multiplier']
        * data['vehicle_type_multiplier']
        * data['location_category_multiplier']
    )

    # Ensure dynamic prices are positive
    data['dynamic_price'] = np.maximum(data['dynamic_price'], 0)  # Avoid negative prices

    return data
# Main preprocessing function
def preprocess_data(data, demand_elasticity, supply_elasticity):
    """
    Preprocess the input data and calculate dynamic prices.

    Parameters:
        data (pd.DataFrame): Raw input data.
        demand_elasticity (float): Estimated demand elasticity.
        supply_elasticity (float): Estimated supply elasticity.

    Returns:
        tuple: Split training and testing sets (X_train, X_test, y_train, y_test).
    """
    # Perform preprocessing steps (e.g., scaling, encoding)
    # Adjust these according to your preprocessing requirements
    feature_list = ['Number_of_Drivers', 'Customer_Loyalty_Status','Number_of_Riders',
                    'Time_of_Booking','Vehicle_Type',
                     'Expected_Ride_Duration','Location_Category']
    target_variable = 'dynamic_price'

    # Preprocessing pipeline setup
    categorical_features = [feature for feature in feature_list if data[feature].dtype == 'object']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])
    
    # Apply dynamic price calculation
    new_data = calculate_dynamic_price(data, demand_elasticity, supply_elasticity)
    
    # Transform the features for modeling
    X = new_data[feature_list]
    y = new_data[target_variable]
    X_processed = pipeline.fit_transform(X)
    print(X.columns)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

    return pipeline, X_train, X_test, y_train, y_test
