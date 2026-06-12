"""
# Projet MLOps - Prédiction de pluie en Australie
==============================

Objectif : prédire la pluie de demain

Les tâches :

- Ré-utilisation d'un dataset provenant d'un projet précédent (weatherAUS_encoded.csv), dont les données sont prétraitées
- Entraînement et inférence d'un modèle ML : Random Forest
- Suivi des expériences avec MLflow + enregistrementes des Runs
- Versionnement des datasets avec DVC
- API d'inférence développée avec FastAPI
- Authentification avec Basic Auth (OAuth2 ayant été étudié)
- Dockerisation
- Intégration et déploiement continus (CI/CD) avec GitHub Actions
- Monitoring des performances avec Prometheus & Grafana
- Interface utilisateur avec Streamlit

Attention : l'arborescence se base sur une structure initiale existante, seuls les élèments marqués <- ont été utilisés.

Arborescence des fichiers

.
├── Dockerfile                             <- Image Docker de l'API FastAPI
├── Dockerfile.streamlit                   <- Image Docker de l'application Streamlit
├── LICENSE
├── README.md
├── data
│   └── processed
│       ├── weatherAUS_encoded.csv         <- Dataset du projet
│       └── weatherAUS_encoded.csv.dvc     <- Versionnement DVC
│
├── docker-compose.yml                     <- Orchestration des services Docker
├── init_predictions.py                    <- Script d'initialisation (Entrainement du modèle et lancement de prédictions)
├── monitoring
│   ├── grafana
│   │   └── dashboards
│   └── prometheus
│       └── prometheus.yml                 <- Configuration Prometheus
│
├── notebooks
├── pytest.ini
├── references
├── reports
│   └── figures
│
├── requirements.txt                       <- Installation des Dépendances Python
│
├── src
│   ├── __init__.py
│   │
│   ├── config
│   │
│   ├── data
│   │   ├── load_postgres.py               <- Chargement des données dans PostgreSQL (avec weatherAUS encoded.csv)
│   │   └── make_dataset.py
│   │
│   ├── features
│   │   └── build_features.py
│   │
│   ├── main.py                            <- FastAPI sans authentification
│   ├── main_auth.py                       <- FastAPI avec authentification Basic Auth (celui utilisé durant tout le projet)
│   └── main_OAuth2.py                     <- FastAPI avec authentification OAuth2
│   │
│   ├── models
│   │   ├── training.py                    <- Entraînement d'un modèle Random Forest sur les données
│   │   └── predict.py                     <- Prédictions données par ce modèle
│   │
│   ├── visualization
│       └── visualize.py
│
├── streamlit_app.py                       <- Interface Streamlit pour utiliser l'API
│
└── tests                                  <- Tests Unitaires
    ├── test_load_postgres.py              <- Tests sur le chargement des données
    ├── test_main.py                       <- Tests de l'API sans Authentificaition
    ├── test_main_auth.py                  <- Tests de l'API avec Authentification Basic Auth
    ├── test_predict.py                    <- Tests des prédictions
    └── test_training.py                   <- Tests de l'entrainement du modèle



Architecture MLOPS


┌─────────────────────────────┐
│ Données prétraitées         │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────────────────────┐
│ Chargement des données dans PostgreSQL      │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│ Training (training.py)                      │
│ • MLflow Tracking                           │
│ • DVC Versioning                            │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│ MLflow Registry                             │
│ • Comparaison des modèles                   │
│ • Promotion du meilleur modèle              │
└──────────────┬──────────────────────────────┘
               ▼
┌────────────────────────────────────────────--─┐
│ API FastAPI                                   │
│ • /training   • /predict   • /webhook/grafana │
└──────────────┬─────────────────────────────--─┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ Monitoring Prometheus & Grafana             │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│ Airflow                                     │
│ • Détection de drift (Evidently)            │
│ • Retraining automatique                    │
│ • Orchestration pipelines ML                │
└──────────────┬──────────────────────────────┘
               ▼
      ┌──────────────────┐
      │    Streamlit     │
      └──────────────────┘

--------
"""
