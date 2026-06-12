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
┌─────────────────────────────────────────┐
│ FastAPI                                 │
│ /predict   /training  /webhook/grafana  │
└──────────────┬──────────────────────────┘
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
│ Streamlit                  │
└─────────────────────────────┘
```
