from typing import Dict
import subprocess
import sys
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import predict


# --- CONFIGURATION DES CHEMINS ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = BASE_DIR / "src" / "models" / "training.py"


# --- INITIALISATION API ---------------------------------------------------------------
app = FastAPI(
    title="Weather Prediction API",
    description="API d'entraînement et d'inférence pour la prédiction de pluie en Australie.",
    version="1.0.0"
)


# --- SCHEMA D'ENTREE POUR LA PREDICTION ----------------------------------------------
class WeatherInput(BaseModel):
    Humidity3pm: float = Field(..., example=55.0)
    Humidity9am: float = Field(..., example=70.0)
    Rainfall: float = Field(..., example=0.0)
    WindGustSpeed: float = Field(..., example=35.0)
    Pressure3pm: float = Field(..., example=1012.0)
    MaxTemp: float = Field(..., example=24.0)
    Temp3pm: float = Field(..., example=22.0)
    Year: int = Field(..., example=2016)
    Month: int = Field(..., example=6)


# --- ROUTE DE BASE -------------------------------------------------------------------
@app.get("/")
def read_root() -> Dict[str, str]:
    return {
        "message": "Weather Prediction API is running",
        "documentation": "/docs"
    }


# --- ENDPOINT PREDICTION -------------------------------------------------------------
@app.post("/predict")
def predict_rain(data: WeatherInput) -> Dict:
    try:
        input_data = data.model_dump()
        return predict(input_data)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(error)}"
        )


# --- ENDPOINT TRAINING ---------------------------------------------------------------
@app.post("/training")
def train_model() -> Dict:
    try:
        result = subprocess.run(
            [sys.executable, str(TRAINING_SCRIPT)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BASE_DIR),
            check=False,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",  # corrige l'erreur MLflow sous Windows
                "PYTHONUTF8": "1"
            }
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Erreur lors de l'entraînement du modèle.",
                    "stderr": result.stderr
                }
            )

        return {
            "message": "Entraînement terminé avec succès.",
            "stdout": result.stdout
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur inattendue lors de l'entraînement : {str(error)}"
        )