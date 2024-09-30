from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
from src.config import MODEL_PATH, PIPELINE_PATH  # Update to include the pipeline path
import pandas as pd
import pickle

# Define the structure of incoming prediction requests
class PricingRequest(BaseModel):
    Number_of_Drivers: int
    Customer_Loyalty_Status: str
    Time_of_Booking: str
    Expected_Ride_Duration: int
    Number_of_Riders: int
    Vehicle_Type: str
    Location_Category: str
    # Include all relevant features here

# Initialize the FastAPI app
app = FastAPI()

# Load the model and pipeline once at startup
@app.on_event("startup")
def load_model():
    global model, pipeline
    # Load the model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Load the preprocessor pipeline
    pipeline = load(PIPELINE_PATH)  # Load the pre-fitted pipeline (includes transformer and scaler)

# Define the prediction endpoint
@app.post("/predict")
def get_prediction(request: PricingRequest):
    try:
        # Convert request data to a DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Apply the pre-fitted pipeline to transform the input data
        transformed_data = pipeline.transform(input_data)

        # Generate prediction
        prediction = model.predict(transformed_data)

        # Return prediction as a response
        return {"predicted_price": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
