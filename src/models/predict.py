from typing import Dict, List

import mlflow.sklearn
import pandas as pd
import os
import mlflow

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
)

MODEL_NAME = "WeatherRandomForest"
MODEL_STAGE = "Production"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

FEATURES: List[str] = [
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
    """
    Charge le modèle actuellement promu en Production dans le Model Registry MLflow.
    """
    return mlflow.sklearn.load_model(MODEL_URI)


def validate_input(data: Dict) -> None:
    """
    Vérifie que toutes les variables nécessaires à la prédiction sont présentes.
    """
    missing_features = [feature for feature in FEATURES if feature not in data]

    if missing_features:
        raise ValueError(
            f"Variables manquantes pour la prédiction : {missing_features}"
        )


def prepare_input(data: Dict) -> pd.DataFrame:
    """
    Convertit les données d'entrée en DataFrame avec l'ordre exact des features attendu par le modèle.
    """
    validate_input(data)

    input_df = pd.DataFrame([data])
    input_df = input_df[FEATURES]

    return input_df


def predict(data: Dict) -> Dict:
    """
    Réalise une prédiction à partir d'un dictionnaire de variables météo.
    """
    model = load_model()
    input_df = prepare_input(data)

    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction),
        "label": "RainTomorrow" if int(prediction) == 1 else "NoRainTomorrow",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE
    }


if __name__ == "__main__":
    sample_data = {
        "Humidity3pm": 55.0,
        "Humidity9am": 70.0,
        "Rainfall": 0.0,
        "WindGustSpeed": 35.0,
        "Pressure3pm": 1012.0,
        "MaxTemp": 24.0,
        "Temp3pm": 22.0,
        "Year": 2016,
        "Month": 6
    }

    result = predict(sample_data)
    print(result)