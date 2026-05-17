# FAST API LAUNCHER

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# MODEL AND PREPROCESSORS PATH
MODEL_PATH = Path("./model/trained_model.pkl")
PREPROCESSORS_PATH = Path("./preprocessors/preprocessors.pkl")

MODEL = joblib.load(MODEL_PATH)
PREPROCESSORS = joblib.load(PREPROCESSORS_PATH)

# DEFINE THE INPUT SCHEMA
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# PREPROCESSING THE INPUT DICTIONARY
def preprocessing_user_input(payload: dict):
    """
    ACCEPTS RAW JSON DICT, CONVERT TO DATAFRAME, AND APPLIES EACH PREPROCESSOR FROM THE SAVED DISCTIONARY.
    """

    # CONVERT JSON TO DATAFRAME
    df = pd.DataFrame([payload])

    # APPLYING EACH PREPROCESSOR IN ORDER
    for name, preprocessor in PREPROCESSORS.items():
        df = pd.DataFrame(preprocessor.transform(df), columns=df.columns)

    return df.values


# PREDICTION FUNCTION
def predicting_diabetes(payload: dict):
    labels = np.array(["No_Diabetes", "Has_Diabetes"])

    preprocessed = preprocessing_user_input(payload)
    predictions = MODEL.predict(preprocessed)[0]

    return int(predictions), labels[predictions]


# FASTAPI APP
app = FastAPI(title="DIABETES STATUS DETERMINANT")

@app.get("/")
def home():
    return {"message": "THE API IS RUNNING..."}



@app.post("/predict")
async def predict(request: DiabetesInput):
    payload = request.dict()

    predictions, status = predicting_diabetes(payload)

    return {
        "PREDICTION": predictions,
        "STATUS": status
    }

