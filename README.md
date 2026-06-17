# Projet MLOps - Prédiction de pluie en Australie
==============================

Objectif : prédire la pluie de demain

Les tâches :
- Ré-utilisation d'un dataset provenant d'un projet précédent (weatherAUS_encoded.csv)
- Entraînement d'un modèle ML : Random Forest
- MLflow + tracking des runs
- Versionnement des datasets avec DVC
- API FastAPI
- Docker + CI/CD GitHub Actions
- Monitoring Prometheus & Grafana
- Interface Streamlit

---

# Architecture MLOPS

```text
┌─────────────────────────────┐
│     Données prétraitées     │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Chargement dans PostgreSQL  │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Training (MLflow + DVC)     │
└──────────────┬──────────────┘
               ▼
┌──────────────────────────────────────┐
│ MLflow Registry                      │
│ Comparaison et promotion des modèles │
└──────────────┬───────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│ FastAPI                                    │
│ -/predict   -/training  -/webhook/grafana  │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────┐
│ Monitoring Prometheus + Grafana        │
└──────────────┬─────────────────────────┘
               ▼
┌─────────────────────────────────┐
│ Airflow                         │
│ Détection de drift (Evidently)  │
│ Re-entrainement automatique y)  │
│ Orchestration pipelines ML      │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────┐
│          Streamlit          │
└─────────────────────────────┘
```

---

# Instructions d'installation
- Cloner le repo : git clone "<url-du-repository>"
- Lancer les composants avec Docker suivant la commande : docker compose up -d --build

Vérifier les conteneurs : docker ps (Cette vérification peut se faire via Streamlit - Page 'API Status', indiquant les états fonctionnels dess composants)

Initialisation des données : lancer init_predictions.py pour un premier entraînement de modèle et l'envoi de prédictions, afin d'alimenter l'API
- python3 init_predictions.py
(ce fichier python lance src/training.py)

# TESTS
TESTS : il possible de tester une prédiction en ligne de commande sous Ubuntu
curl -X POST http://localhost:8000/predict \
-u admin:password \
-H "Content-Type: application/json" \
-d '{
    "Humidity3pm": 55,
    "Humidity9am": 70,
    "Rainfall": 0,
    "WindGustSpeed": 35,
    "Pressure3pm": 1012,
    "MaxTemp": 24,
    "Temp3pm": 22,
    "Year": 2020,
    "Month": 6
}'

# Accès Streamlit
Accès à Streamlit : http://localhost:8501
Le Monitoring est directement accessible depuis streamlit, incluant Prometheus & Grafana (http://localhost:3000)

# Lancement des Tests unitaires :
depuis la racine du projet : pytest
