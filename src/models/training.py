# Importation des modules
import os
import numpy as np
import joblib
import pandas as pd
import yaml

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sqlalchemy import create_engine


# --- DEFINITION DES VARIABLES ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

TABLE_NAME = "weather_data"
MODEL_PATH = MODELS_DIR / "random_forest_model.pkl"

EXPERIMENT_NAME = "weather_prediction"
MODEL_NAME = "WeatherRandomForest"

FEATURES = [
    "Humidity3pm",
    "Humidity9am",
    "Rainfall",
    "WindGustSpeed",
    "Pressure3pm",
    "MaxTemp",
    "Temp3pm",
    "Year",
    "Month",
]


# --- CONFIGURATION POSTGRESQL ET MLFLOW -----------------------------------------------

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "admin")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "weather_db")

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# --- FONCTIONS ------------------------------------------------------------------------

def add_noise(df, features, noise_level=0.05):
    """Ajoute un bruit gaussien aux variables numériques."""
    df_noisy = df.copy()

    for col in features:
        df_noisy[col] += np.random.normal(
            0,
            noise_level * df[col].std(),
            size=len(df),
        )

    return df_noisy


def load_data():
    """Charge les données depuis PostgreSQL."""
    query = f"SELECT * FROM {TABLE_NAME}"
    df = pd.read_sql(query, engine)
    return df


def get_dvc_data_version():
    """Récupère le hash MD5 du dataset versionné par DVC."""
    dvc_file = (
        BASE_DIR.parent
        / "data"
        / "processed"
        / "weatherAUS_encoded.csv.dvc"
    )

    with open(dvc_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data["outs"][0]["md5"]


# --- MAIN -----------------------------------------------------------------------------

def main():
    df = load_data()
    dvc_version = get_dvc_data_version()
    df = add_noise(df, FEATURES, noise_level=0.05)

    X = df[FEATURES]
    y = df["RainTomorrow"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        print("F1-score :", round(f1, 4))
        print(classification_report(y_test, y_pred))

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("data_version_dvc", dvc_version)

        mlflow.log_metric("f1_score", f1)
        mlflow.log_text(
            classification_report(y_test, y_pred),
            "classification_report.txt",
        )

        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME,
        )

        joblib.dump(model, MODEL_PATH)

    client = MlflowClient()

    models_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = max(models_versions, key=lambda v: int(v.version))

    try:
        model_prod = mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}/Production"
        )

        y_pred_old = model_prod.predict(X_test)
        old_f1 = f1_score(y_test, y_pred_old)

        print(f"Ancien modèle Production F1 : {round(old_f1, 4)}")
        print(f"Nouveau modèle F1 : {round(f1, 4)}")

        if f1 > old_f1:
            print("Le nouveau modèle est meilleur. Promotion en Production.")

            for version in models_versions:
                if version.current_stage == "Production":
                    client.transition_model_version_stage(
                        name=MODEL_NAME,
                        version=version.version,
                        stage="Archived",
                    )

            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest_version.version,
                stage="Production",
            )
        else:
            print("Le modèle en Production reste meilleur.")

    except Exception:
        print("Aucun modèle en Production. Promotion automatique.")

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version.version,
            stage="Production",
        )


if __name__ == "__main__":
    main()