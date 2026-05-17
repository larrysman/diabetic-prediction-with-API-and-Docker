# STAGE 03: PREPROCESSING STAGE

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from imblearn.combine import SMOTEENN

from src.load_data import loading_data


# PREPROCESSING OF THE DATASET
def prep_pipeline(df: pd.DataFrame, target_col : str):
    # INITIALIZE THE PREPROCESSORS
    std = StandardScaler()
    min_max = MinMaxScaler()
    norm = Normalizer()
    smote_enn = SMOTEENN(random_state=42)

    target = df[target_col]
    features = df.drop(columns=[target_col])

    print(f"Before: {df[target_col].value_counts()}")

    # APPLY PREPROCESSORS
    num_std = std.fit_transform(features)
    num_min = min_max.fit_transform(num_std)
    num_norm = norm.fit_transform(num_min)

    num_df = pd.DataFrame(num_norm, columns=features.columns)
    tgt_df = target.reset_index(drop=True)

    # CORRECTING THE DATA IMBALANCE
    num_df, tgt_df = smote_enn.fit_resample(num_df, tgt_df)

    # FINAL PREPROCESSED DATAFRAME
    prep_df = pd.concat([num_df, tgt_df], axis=1)

    print(f"After: {prep_df[target_col].value_counts()}")

    # SAVING THE PREPROCESSORS
    PREPROC_DIR = "preprocessors"
    os.makedirs(PREPROC_DIR, exist_ok=True)

    PREPROCESSORS = {
        "STANDARD_SCALER": std,
        "MIN_MAX_SCALER": min_max,
        "NORMALIZE_SCALER": norm
    }

    preproc_path = os.path.join(PREPROC_DIR, "preprocessors.pkl")
    joblib.dump(PREPROCESSORS, preproc_path)

    print(f"Preprocessors saved to: {preproc_path}")

    return prep_df, std, min_max, norm


if __name__ == "__main__":

    data_path = "./data/diabetes.csv"
    df = loading_data(data_path)

    prep_df, std_scaler, min_max_scaler, norm_scaler = prep_pipeline(df, "Outcome")