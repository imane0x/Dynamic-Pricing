# src/data_loader.py

import pandas as pd
from config import DATA_PATH

def load_data():
    """
    Loads the dataset.

    Returns:
        DataFrame: Raw data.
    """
    df = pd.read_csv(DATA_PATH)
    return df
