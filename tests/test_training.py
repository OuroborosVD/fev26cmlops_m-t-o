# Importation des modules
import pandas as pd
import numpy as np
import pytest

from unittest.mock import patch, MagicMock
from src.models.training import add_noise, get_dvc_data_version, main, FEATURES


def test_add_noise():
    """ Vérifier l'ajout du bruit """

    df = pd.DataFrame({
        "Humidity3pm": [10, 20],
        "Humidity9am": [30, 40],
        "Rainfall": [1, 2],
        "WindGustSpeed": [5, 6],
        "Pressure3pm": [1000, 1001],
        "MaxTemp": [25, 26],
        "Temp3pm": [20, 21],
        "Year": [2020, 2021],
        "Month": [5, 6]
    })

    df_noisy = add_noise(df, FEATURES)

    assert df_noisy.shape == df.shape
    assert not df_noisy.equals(df)  # les valeurs doivent changer


def test_get_dvc_data_version():
    """ Vérifie la lecture du fichier DVC """

    fake_yaml = {
        "outs": [
            {"md5": "123abc"}
        ]
    }

    with patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=fake_yaml):

        version = get_dvc_data_version()

        assert version == "123abc"


def test_main_training_pipeline():
    """ Test du pipeline d'entraînement sans dépendances externes"""

    fake_df = pd.DataFrame({
        "Humidity3pm": [10, 20, 30, 40],
        "Humidity9am": [30, 40, 50, 60],
        "Rainfall": [1, 2, 3, 4],
        "WindGustSpeed": [5, 6, 7, 8],
        "Pressure3pm": [1000, 1001, 1002, 1003],
        "MaxTemp": [25, 26, 27, 28],
        "Temp3pm": [20, 21, 22, 23],
        "Year": [2020, 2021, 2022, 2023],
        "Month": [5, 6, 7, 8],
        "RainTomorrow": [0, 0, 1, 1]
    })

    mock_model = MagicMock()
    mock_model.predict.return_value = [1, 0]

    with patch("src.models.training.load_data", return_value=fake_df), \
         patch("src.models.training.add_noise", return_value=fake_df), \
         patch("src.models.training.MlflowClient"), \
         patch("src.models.training.mlflow.start_run"), \
         patch("src.models.training.mlflow.sklearn.log_model"), \
         patch("src.models.training.mlflow.log_param"), \
         patch("src.models.training.mlflow.log_metric"), \
         patch("src.models.training.mlflow.log_text"), \
         patch("src.models.training.mlflow.sklearn.load_model", return_value=mock_model):

        # On vérifie juste que le pipeline s'exécute sans erreur
        main()
           
