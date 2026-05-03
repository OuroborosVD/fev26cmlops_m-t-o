# Importation de modules
from fastapi.testclient import TestClient  # Simulation de requêtes HTTP sans serveur réel
from unittest.mock import patch  # Permet de remplacer temporairement des fonctions (mock)
from src.main import app

# Objectif : tester l’API sans réseau réel

# Instanciation d'un client
client = TestClient(app)

# Credentials utilisés pour les endpoints protégés
AUTH = ("admin", "password")


# Instanciation de données valides
def get_valid_payload():
    """Retourne un exemple de données météo valides"""
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
    """Test du endpoint / donc de l'API """

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Weather Prediction API is running"
    assert data["documentation"] == "/docs"


# Test de Predict par une fausse fonction (pour éviter le vrai chargement du ML)
def test_predict_success():
    """Test du endpoint /predict avec succès"""

    mock_response = {"prediction": 1}

    # avec patch, on remplace temporairement la fonction predict, pour tester l'API et non le modèle
    # Pour rendre le test rapide fiable    
    with patch("src.main.predict", return_value=mock_response):

        response = client.post(
            "/predict",
            json=get_valid_payload(),
            auth=AUTH  # AUTH ajoutée pour sécuriser l'endpoint
        )

        assert response.status_code == 200
        assert response.json() == mock_response


# Simulation d'une erreur volontairement
def test_predict_failure():
    """ Test du endpoint "predict" en cas d'échec """    

    # On simule une erreur dans la fonction predict avec patch
    with patch("src.main.predict", side_effect=Exception("boom")):

        response = client.post(
            "/predict",
            json=get_valid_payload(),
            auth=AUTH  # AUTH 
        )

        assert response.status_code == 500
        assert "Erreur lors de la prédiction" in response.json()["detail"]


# Test du training (avec succès) ------------------------------------------------------
def test_training_success():
    """Test du endpoint /training en cas de succès"""

    # On crée un faux résultat
    mock_result = type("obj", (object,), {
        "returncode": 0,
        "stdout": "training ok",
        "stderr": ""
    })

    # On remplace par ce faux résultat
    with patch("src.main.subprocess.run", return_value=mock_result):

        response = client.post(
            "/training",
            auth=AUTH  # AUTH
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Entraînement terminé avec succès."


# Test du training (avec échec) ------------------------------------------------------
def test_training_failure():
    """ Test du endpoint /training en cas d'échec """

    # On simule un échec du script d'entraînement
    mock_result = type("obj", (object,), {
        "returncode": 1,
        "stdout": "",
        "stderr": "error"
    })

    # On mock subprocess.run
    with patch("src.main.subprocess.run", return_value=mock_result):

        response = client.post(
            "/training",
            auth=AUTH  # AUTH
        )

        assert response.status_code == 500
        assert "Erreur lors de l'entraînement" in response.json()["detail"]["message"]


# --- TEST SANS AUTHENTIFICATION (IMPORTANT) -------------------------------------------
def test_predict_without_auth():
    """Vérifie que l'accès sans authentification est refusé"""

    response = client.post(
        "/predict",
        json=get_valid_payload()
    )

    assert response.status_code == 401
