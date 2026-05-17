# STAGE 01: DATA LOADING

import pandas as pd
import numpy as np

def loading_data(path: str) -> pd.DataFrame:
    """
    THIS FUNCTION LOADS THE DATASET FROM DATA/ FOR FURTHER PROCESSSING.

    Args:
        Path | str: THE PATH TO THE DATASET

    Returns:
        pd.DataFrame: THE LOADED DATAFRAME
    """

    return pd.read_csv(path)


if __name__ == "__main__":
    data_path = "./data/diabetes.csv"
    df = loading_data(data_path)
    print(df.head())
    #print(df.describe())
    print(f"The loaded dataset has {df.shape[0]} rows and {df.shape[1]} columns.\nThe columns are {list(df.columns)}")