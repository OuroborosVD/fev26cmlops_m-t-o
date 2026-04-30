# Importation de modules
from fastapi.testclient import TestClient # Simulation de requêtes HTTP
from unittest.mock import patch # Permet de mocker une fonction
from src.main import app

# Objectif : Test l'API sans réseau réel

# Instanciation d'un client
client = TestClient(app)

# Instanciation de données valides
def get_valid_payload():
    return {
        "Humidity3pm": 55.0,
        "Humidity9am": 70.0,
        "Rainfall": 0.0,
        "WindGustSpeed": 35.0,
        "Pressure3pm": 1012.0,
        "MaxTemp": 24.0,
        "Temp3pm": 22.0,
        "Year": 2016,
        "Month": 6
    }


# Test de Santé (check endpoint)
def test_endpoint_origin():
    """ Test du endpoint d'origine / donc de l'API"""

    # Requête GET
    response = client.get("/")

    # Validation de la réponse HTTP
    assert response.status_code == 200

    # On récupère le JSON retourné par l'API
    data = response.json()

    # Vérifie le contenu de la réponse
    assert data["message"] == "Weather Prediction API is running"
    assert data["documentation"] == "/docs"


# Test de Predict par une fausse fonction (éviter le vrai chargement du ML)
def test_predict_success():
    """ Test du endpoint "predict" en cas de succès """

    # Réponse fictive de predict si c'est réussi
    mock_response = {"prediction": 1}

    # avec patch, on remplace temporairement la fonction predict, pour tester l'API et non le modèle
    # Pour rendre le test rapide fiable    
    with patch("src.main.predict", return_value=mock_response):

        # Requête POST avec des données météo valides
        response = client.post("/predict", json=get_valid_payload())

        # Validation de la réponse HTTP
        assert response.status_code == 200

        # Comparaison des réponses
        assert response.json() == mock_response


# Simulation d'une erreur volontairement
def test_predict_failure():
    """ Test du endpoint "predict" en cas d'échec """    

    # On simule une erreur dans la fonction predict avec patch
    with patch("src.main.predict", side_effect=Exception("boom")):

        # Requête POST avec des données météo valides
        response = client.post("/predict", json=get_valid_payload())

        # Erreur de la réponse HTTP
        assert response.status_code == 500

        # Vérifie que le message d'erreur est correct
        assert "Erreur lors de la prédiction" in response.json()["detail"]


def test_training_success():
    """ Test du endpoint "training" en cas de succès """

    # On crée un faux résultat
    mock_result = type("obj", (object,), {
        "returncode": 0,        # 0 = succès
        "stdout": "training ok",
        "stderr": ""
    })

    # On remplace par ce faux résultat
    with patch("src.main.subprocess.run", return_value=mock_result):

        # Appel de l'endpoint /training
        response = client.post("/training")

        # Vérifie que l'entraînement est considéré comme réussi
        assert response.status_code == 200

        # Vérifie le message retourné
        assert response.json()["message"] == "Entraînement terminé avec succès."


def test_training_failure():
    """ Test du endpoint "training" en cas d'échec """

    # On simule un échec du script d'entraînement
    mock_result = type("obj", (object,), {
        "returncode": 1,        # ≠ 0 = erreur
        "stdout": "",
        "stderr": "error"
    })

    # On mock subprocess.run
    with patch("src.main.subprocess.run", return_value=mock_result):

        # Appel de l'API
        response = client.post("/training")

        # L'API doit renvoyer une erreur
        assert response.status_code == 500

        # Vérifie que le message d'erreur est correct
        assert "Erreur lors de l'entraînement" in response.json()["detail"]["message"]
