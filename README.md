# Dynamic Pricing Project

## Overview

A project to implement dynamic pricing strategies for ride-sharing platforms using historical data and machine learning.

## Features

- Predictive pricing model
- Real-time price adjustments
- Customer behavior analysis

## Data

- **Source:** [Kaggle](https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset)
- **Key Features:**
  - Riders and drivers count
  - Location type (urban, suburban, rural)
  - Customer loyalty status
  - Booking time and vehicle type
  - Historical pricing data

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/imane0x/Dynamic-Pricing
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
```bash
python main.py
```

4. Build and run the Docker container:
```bash
docker build -t dynamic-pricing .
docker run -p 8000:8000 dynamic-pricing
```   

## Hyperparameter Tuning

Uses Grid Search for tuning the Random Forest Regressor.

## Weights & Biases

Integrated with wandb for experiment tracking. Set up your wandb account and API key.

---
