# FINAL ORCHESTRATOR PIPELINE

import numpy as np
import pandas as pd
import os
from pathlib import Path
import joblib

from src.load_data import loading_data

from src.preprocessing_03 import prep_pipeline

from src.model_training_04 import model_training

# MODEL AND PREPROCESSORS PATH
MODEL_PATH = Path("./model/trained_model.pkl")
PREPROCESSORS_PATH = Path("./preprocessors/preprocessors.pkl")

MODEL = joblib.load(MODEL_PATH)
PREPROCESSORS = joblib.load(PREPROCESSORS_PATH)


# Pregnancies: float,
# Glucose: float,
# BloodPressure: float,
# SkinThickness: float,
# Insulin: float,
# BMI: float,
# DiabetesPedigreeFunction: float,
# Age: int

def inference_pipeline():
    
    # COLLECTING USER'S INPUTS
    Pregnancies = float(input("Enter Pregnancies: "))
    Glucose = float(input("Enter Glucose level: "))
    BloodPressure = float(input("Enter Your Blood Pressure: "))
    SkinThickness = float(input("Enter Skin Thickness number: "))
    Insulin = float(input("Enter your Insulin level: "))
    BMI = float(input("Enter your Body Mass Index(BMI) Number: "))
    DiabetesPedigreeFunction = float(input("Enter Diabetes Pedigree function: "))
    Age = int(input("Enter your current age: "))
    
    # CONVERT INPUTS TO ARRAY
    INPUT_ARRAY = np.array([
        [
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]
    ])

    # CONVERT INPUT TO DATAFRAME
    INPUT_DF = pd.DataFrame(
        INPUT_ARRAY,
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # PREPROCESSING THE INPUTS => FEATURES
    for name, preprocessor in PREPROCESSORS.items():
        prep_features = pd.DataFrame(preprocessor.transform(INPUT_DF.values), columns=INPUT_DF.columns)

    prep_features_array = prep_features.values

    # PREDICTIONS
    LABELS = np.array(["NOT_DIABETIC", "IS_DIABETIC"])

    PREDICTIONS = MODEL.predict(prep_features)

    PRED_DF = pd.DataFrame([PREDICTIONS], columns=["PREDICTED"])
    STATUS = LABELS[PREDICTIONS[0]]
    STATUS_DF = pd.DataFrame([STATUS], columns=["DIABETIC_STATUS"])

    FINAL_DF = pd.concat([PRED_DF, STATUS_DF], axis=1)

    print(FINAL_DF)

    return FINAL_DF


if __name__ == "__main__":
    inference_pipeline()
