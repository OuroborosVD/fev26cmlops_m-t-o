from pathlib import Path
import sqlite3
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DB_PATH = DATA_DIR / "weather.db"
TABLE_NAME = "weather_data"
MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"

#  VARIABLES UTILISÉES POUR L'ENTRAÎNEMENT DU MODÈLE
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


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    X = df[FEATURES]
    y = df["RainTomorrow"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("F1-score :", round(f1_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()