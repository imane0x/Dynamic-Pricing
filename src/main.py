# src/main.py

import pandas as pd
import wandb
from preprocessing import preprocess_data, calculate_dynamic_price
from model import train_model, evaluate_model
from dynamic_pricing import estimate_demand_elasticity, estimate_supply_elasticity
from config import DATA_PATH
import pickle
def main():
    # Initialize wandb with your project name
    wandb.init(project="dynamic_pricing", entity="im21", config={
        "dataset": "Dynamic Pricing Dataset",
        "model": "Random Forest Regressor",
        "test_size": 0.3,
        "random_state": 42,
    })

    # Load data
    data = pd.read_csv(DATA_PATH)
    wandb.log({"dataset_size": len(data)})

    # Estimate elasticities
    demand_elasticity, _ = estimate_demand_elasticity(data)
    supply_elasticity, _ = estimate_supply_elasticity(data)
    new_data = calculate_dynamic_price(data, demand_elasticity, supply_elasticity)
    # Preprocess the data and calculate dynamic prices
    pipeline, X_train, X_test, y_train, y_test = preprocess_data(new_data, demand_elasticity, supply_elasticity)
    wandb.log({
        "demand_elasticity": demand_elasticity,
        "supply_elasticity": supply_elasticity
    })
    print(len(X_train))
    # Train the model
    model = train_model(X_train, y_train)

    # Log model parameters
    wandb.log({"model_params": model.get_params()})

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    # Log evaluation metrics
    wandb.log(metrics)
        # Save the model as a pickle file
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
