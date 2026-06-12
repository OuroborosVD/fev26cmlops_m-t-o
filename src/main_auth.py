# Importation des modules
from typing import Dict
import subprocess
import sys
import os
from pathlib import Path
import secrets
import requests
import threading

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from fastapi import Request
import json

from src.models.predict import predict

# Importation de Prometheus
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import time

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = BASE_DIR / "src" / "models" / "training.py"


# Instanciation FastAPI
app = FastAPI(
    title="Weather Prediction API",
    description="API d'entraînement et d'inférence pour la prédiction de pluie en Australie.",
    version="1.0.0"
)

# Mise en place des métriques pour Prometheus
Instrumentator().instrument(app).expose(app)

# --- CONFIG AUTHENTIFICATION ----------------------------------------------------------
security = HTTPBasic()

# Métriques Prometheus personnalisées
weather_predictions_total = Counter("weather_predictions_total", "Nombre total de predictions")
weather_success_predictions_total = Counter("weather_success_predictions_total", "Nombre de prédictions réussies")
weather_rain_predictions_total = Counter("weather_rain_predictions_total", "Nombre de predictions pluie")
weather_norain_predictions_total = Counter("weather_norain_predictions_total", "Nombre de predictions sans pluie")
weather_prediction_errors_total = Counter("weather_prediction_errors_total", "Nombre d erreurs de prediction")
weather_prediction_duration_seconds = Histogram("weather_prediction_duration_seconds", "Temps de prediction")


# Récupération des credentials depuis les variables d’environnement
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "password")

# Fonction d'authentification Basic
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    """ Vérifie les identifiants, compare_digest évite certaines attaques """
    correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)

    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})

    return credentials.username

# L’authentification admin sert à éviter l'accès provenant d'un utilisateur lambda
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")

def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """ Restreindre certains endpoints, seul l'utilisateur 'admin' est autorisé """
    username = credentials.username
    password = credentials.password

    if not (
        secrets.compare_digest(username, ADMIN_USERNAME)
        and secrets.compare_digest(password, API_PASSWORD)
    ):
        raise HTTPException(status_code=403, detail="Admin only")

    return username


# Classe d'entrée Weather prédéfinie
class WeatherInput(BaseModel):
    # Validation automatique des entrées
    Humidity3pm: float = Field(..., example=55.0)
    Humidity9am: float = Field(..., example=70.0)
    Rainfall: float = Field(..., example=0.0)
    WindGustSpeed: float = Field(..., example=35.0)
    Pressure3pm: float = Field(..., example=1012.0)
    MaxTemp: float = Field(..., example=24.0)
    Temp3pm: float = Field(..., example=22.0)
    Year: int = Field(..., example=2016)
    Month: int = Field(..., example=6)


# Endpoint de base
@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Weather Prediction API is running", "documentation": "/docs"}

# Endpoint Initialisation
@app.post("/init")
def init_data():
    try:
        subprocess.run([sys.executable, "init_predictions.py"], check=True)
        return {"message": "Initialisation réussie"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint /predict (avec Authentification !!)
@app.post("/predict")
def predict_rain(
    data: WeatherInput,
    user: str = Depends(authenticate)
) -> Dict:

    start_time = time.time()

    try:
        input_data = data.model_dump()
        result = predict(input_data)

        weather_predictions_total.inc()
        weather_success_predictions_total.inc()

        if result["prediction"] == 1:
            weather_rain_predictions_total.inc()
        else:
            weather_norain_predictions_total.inc()

        return result

    except Exception as error:

        weather_prediction_errors_total.inc()
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(error)}")

    finally:
        weather_prediction_duration_seconds.observe(time.time() - start_time)


# Endpoint /training - entrainement du modèle
@app.post("/training")
def train_model(user: str = Depends(authenticate_admin)) -> Dict:
    try:
        # Lancement du script d'entraînement dans un subprocess
        result = subprocess.run(
            [sys.executable, str(TRAINING_SCRIPT)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BASE_DIR),
            check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        )

        # Gestion des erreurs du script
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail={"message": "Erreur lors de l'entraînement du modèle.", "stderr": result.stderr})

        return {"message": "Entraînement terminé avec succès.", "stdout": result.stdout}

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erreur inattendue lors de l'entraînement : {str(error)}")


# Déclencheur de ré-entrainement
def trigger_training():
    try:
        requests.post("http://api:8000/training", auth=("admin", "password"), timeout=300)
    except Exception as e:
        print("TRAINING ERROR:", e)

# Endpoint Webhook - qui lance un ré-entrainement si détection d'une alerte par grafana
@app.post("/webhook/grafana")
async def grafana_webhook(request: Request):

    data = await request.json()

    print("GRAFANA ALERT:", data)
    print("TRAINING TRIGGERED")

    threading.Thread(target=trigger_training).start()

    return {"status": "received"}
