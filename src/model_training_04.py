# STAGE 04: MODEL TRAINING

import os
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.load_data import loading_data
from src.preprocessing_03 import prep_pipeline

def model_training(df: pd.DataFrame, target_col: str):

    target = df[target_col]
    feature = df.drop(columns=[target_col])

    # SPLIT DATA INTO TRAIN_TEST_SPLIT
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.1, random_state=42)

    # INSTANTIATE MODEL
    MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
    # TRAIN THE MODEL
    MODEL.fit(X_train, y_train)
    # MAKE PREDICTIONS WITH THE MODEL
    Y_PRED = MODEL.predict(X_test)
    # EVALUATE THE MODEL PERFORMANCE
    ACC = accuracy_score(y_test, Y_PRED)
    PREC = precision_score(y_test, Y_PRED)
    REC = recall_score(y_test, Y_PRED)
    F1_SCORE = f1_score(y_test, Y_PRED)

    METRICS = {
        "ACCURACY": ACC,
        "PRECISION": PREC,
        "RECALL": REC,
        "F1_SCORE": F1_SCORE
    }

    # PRINT METRICS
    print("\nMODEL PERFORMANCE METRICS: ")
    for key, value in METRICS.items():
        print(f"{key}: {value:.2f}")

    # SAVING THE MODEL
    MODEL_DIR = "model"
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
    joblib.dump(MODEL, model_path)
    print(f"\nModel saved to: {model_path}")

    # SAVING THE METRICS
    METRICS_DIR = "metrics"
    os.makedirs(METRICS_DIR, exist_ok=True)

    metrics_path = os.path.join(METRICS_DIR, "metrics.json")
    with open(metrics_path, "w") as file:
        json.dump(METRICS, file, indent=4)

    print(f"Metrics saved to: {metrics_path}")

    return MODEL, METRICS


if __name__ == "__main__":
    data_path = "./data/diabetes.csv"
    df = loading_data(data_path)

    prep_df, std_scaler, min_max_scaler, norm_scaler = prep_pipeline(df, "Outcome")
    model, metrics = model_training(prep_df, "Outcome")