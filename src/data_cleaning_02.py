# STAGE 02: DATA CLEANING


import pandas as pd
from src.load_data import loading_data

# data_path = "./data/diabetes.csv"
# df = loading_data(data_path)

# for col in df.columns:
#     if col == "Outcome":
#         continue
#     print(f"==========={col}=======")
#     print("min", df[col].min())
#     print("max", df[col].max())
#     print("average", df[col].mean())
#     print("median", df[col].median())
#     # print("mode", df[col].mode())

def detect_outliers_iqr(df: pd.DataFrame, columns=None):
    """
    Detect outliers in numerical columns using the IQR method.

    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of columns to check. If None, all numeric columns are used.

    Returns:
        dict: A dictionary with column names and number of outliers detected.
    """

    if columns is None:
        columns = df.select_dtypes(include="number").columns

    outlier_summary = {}

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = len(outliers)

    return outlier_summary


if __name__ == "__main__":

    data_path = "./data/diabetes.csv"
    df = loading_data(data_path)

    print(f"The data contains missing values: {df.isna().sum()}")
    print(f"The data has duplicate entries: {df.duplicated().sum()}")

    print("========================================================")

    print(df.Outcome.value_counts())

    results = detect_outliers_iqr(df)
    print("Outlier counts per column:")
    print(results)
