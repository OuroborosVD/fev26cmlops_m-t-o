# importations de moduless
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.models.predict import validate_input, prepare_input, predict, FEATURES


def test_validate_input_ok():
    """ Mise des Features à 1 pour la validation des entrées """
    data = {feature: 1 for feature in FEATURES}

    validate_input(data)


def test_validate_input_missing_features():
    """ Mise des Features à 1 sauf les 2 dernières pour la non-validation des entrées """    
    data = {feature: 1 for feature in FEATURES[:-2]}  # suppression de 2 variables

    with pytest.raises(ValueError) as e:
        validate_input(data)

    assert "Variables manquantes" in str(e.value)


def test_prepare_input():
    """ Vérifier la structure du DataFrame """
    data = {feature: 1 for feature in FEATURES}

    df = prepare_input(data)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == FEATURES
    assert df.shape == (1, len(FEATURES))


def test_predict_success():
    """ Test dans le cas où le modèle prédit la pluie = 1 """

    data = {feature: 1 for feature in FEATURES}

    # Mock du modèle ML
    mock_model = MagicMock() # Instanciation d'un objet factice, à la place du ML (test unitaire <> tets intégration)
    mock_model.predict.return_value = [1]

    # On remplace load_model pour éviter MLflow
    with patch("src.models.predict.load_model", return_value=mock_model):

        result = predict(data)

        assert result["prediction"] == 1
        assert result["label"] == "RainTomorrow"
        assert result["model_name"] == "WeatherRandomForest"
        assert result["model_stage"] == "Production"


def test_predict_no_rain():
    """ Test dans le cas où le modèle prédit autre chose = 0 """

    data = {feature: 1 for feature in FEATURES}

    # Mock du modèle ML
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]

    # On remplace load_model pour éviter MLflow
    with patch("src.models.predict.load_model", return_value=mock_model):

        result = predict(data)

        assert result["prediction"] == 0
        assert result["label"] == "NoRainTomorrow"
        assert result["model_name"] == "WeatherRandomForest"
        assert result["model_stage"] == "Production"
