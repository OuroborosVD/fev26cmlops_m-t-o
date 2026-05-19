# Importations 
import subprocess
import sys
import os

from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import (OAuth2PasswordBearer, OAuth2PasswordRequestForm)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from src.models.predict import predict

# Configuration des chemins 
BASE_DIR = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = BASE_DIR / "src" / "models" / "training.py"


# Clé secrète JWT dans une variable os
SECRET_KEY = os.getenv("SECRET_KEY", "admin")

# Signature du token JWT + durée de validité
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Instanciation API
app = FastAPI(title="Weather Prediction API", description="API sécurisée avec OAuth2")

# Hashage avec bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer attend un endpoint /token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Faux utilisateur
fake_users_db = {"admin": {"username": "admin", "hashed_password": pwd_context.hash("admin"), "role": "admin"},
                 "user": {"username": "user", "hashed_password": pwd_context.hash("user"), "role": "user"}
                }

# Authentifications
def verify_password(plain_password, hashed_password):
    """ Vérifie le mot de passe """
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    """Vérifie qu’un utilisateur existe et que son mot de passe est valide."""
    user = fake_users_db.get(username)

    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """ Création d'un token JWT signé """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Récupération user
def get_current_user(token: str = Depends(oauth2_scheme)):
    """ Décode le token JWT et récupère l’utilisateur courant """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré",
        headers={"WWW-Authenticate": "Bearer"},)

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = fake_users_db.get(username)

    if user is None:
        raise credentials_exception
    return user

# Récupération admin
def get_current_admin(user: dict = Depends(get_current_user)):
    """ Vérifie que l’utilisateur est admin """

    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user

# Classe WeatherInput
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

# Endpoints
# /
@app.get("/")
def root():
    return {"message": "Weather Prediction API with OAuth2 is running", "documentation": "/docs"}

# /token
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """ Endpoint de connexion - Retourne un token JWT si les identifiants sont valides """

    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"], "role": user["role"]}, expires_delta=access_token_expires)

    return {"access_token": access_token, "token_type": "bearer"}

# /predict
@app.post("/predict")
def predict_rain(data: WeatherInput, user: dict = Depends(get_current_user)):
    """ Endpoint de prédiction protégé par JWT """
    try:
        input_data = data.model_dump()
        prediction = predict(input_data)
        return {"user": user["username"], "prediction": prediction}

    except Exception as error:

        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {str(error)}")

# /training
@app.post("/training")
def train_model(user: dict = Depends(get_current_admin)):
    """ Endpoint réservé aux administrateurs """
    try:

        result = subprocess.run(
            [sys.executable, str(TRAINING_SCRIPT)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(BASE_DIR),
            check=False
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail={"message": "Erreur entraînement", "stderr": result.stderr})

        return {"message": "Entraînement terminé avec succès", "stdout": result.stdout}

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erreur inattendue : {str(error)}")
