from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

# mêmes features que training
FEATURES = [
    "Humidity3pm",
    "Humidity9am",
    "Rainfall",
    "WindGustSpeed",
    "Pressure3pm",
    "MaxTemp",
    "Temp3pm",
    "Year",
    "Month"
]


def load_model():
    return joblib.load(MODEL_PATH)


def predict(data: dict):
    model = load_model()

    df = pd.DataFrame([data])

    # Sécurité : ordre des colonnes
    df = df[FEATURES]

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(proba)
    }