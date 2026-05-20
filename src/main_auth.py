# Importation des modules 
from typing import Dict
import subprocess
import sys
import os
from pathlib import Path
import secrets

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from src.models.predict import predict

# Importation de Prometheus
from prometheus_fastapi_instrumentator import Instrumentator


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

# Récupération des credentials depuis les variables d’environnement
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "password")


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

# Endpoint /predict (avec Authentification !!)
@app.post("/predict")
def predict_rain(data: WeatherInput, user: str = Depends(authenticate)) -> Dict:
    try:
        # Conversion du modèle Pydantic en dict
        input_data = data.model_dump()

        # Appel du modèle ML
        return predict(input_data)

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(error)}")

# Endpoint /training
@app.post("/training")
def train_model(user: str = Depends(authenticate_admin)  # Restriction admin) -> Dict:
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
