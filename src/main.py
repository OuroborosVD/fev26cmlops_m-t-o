from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict
import subprocess
from pathlib import Path

app = FastAPI()

class WeatherInput(BaseModel):
    Humidity3pm: float
    Humidity9am: float
    Rainfall: float
    WindGustSpeed: float
    Pressure3pm: float
    MaxTemp: float
    Temp3pm: float
    Year: int
    Month: int


@app.get("/")
def home():
    return {"message": "API météo opérationnelle"}


@app.post("/predict")
def make_prediction(data: WeatherInput):
    return predict(data.model_dump())


@app.post("/training")
def train_model():
    result = subprocess.run(
        ["python", "training.py"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr
    }