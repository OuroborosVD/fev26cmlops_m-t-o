# Importation des modules Airflow
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


# Arguments par défaut du DAG
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# Définition du DAG principal
with DAG(
    dag_id="weather_training_pipeline",
    description="Pipeline Airflow de chargement des données et entraînement du modèle météo",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "weather", "training"],
) as dag:

    # Étape 1 : chargement des données retraitées dans PostgreSQL
    load_data_to_postgres = BashOperator(
        task_id="load_data_to_postgres",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/data/load_postgres.py"
        ),
    )

    # Étape 2 : entraînement du modèle et enregistrement dans MLflow
    train_model_with_mlflow = BashOperator(
        task_id="train_model_with_mlflow",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/models/training.py"
        ),
    )

    # Étape 3 : vérification simple de la disponibilité de l'API
    check_api_status = BashOperator(
        task_id="check_api_status",
        bash_command=(
            "python -c \"import urllib.request; "
            "print(urllib.request.urlopen('http://api:8000/').read().decode())\""
        ),
    )

    # Ordonnancement des tâches
    load_data_to_postgres >> train_model_with_mlflow >> check_api_status