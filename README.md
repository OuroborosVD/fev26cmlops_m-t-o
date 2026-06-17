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
- Cloner le repo : git clone &lt;url-du-repository&gt;
- Lancer les composants avec Docker suivant la commande : docker compose up -d --build

Vérifier les conteneurs : docker ps (Cette vérification peut se faire via Streamlit - Page 'API Status', indiquant les états fonctionnels dess composants)

Initialisation des données : lancer init_predictions.py pour un premier entraînement de modèle et l'envoi de prédictions, afin d'alimenter l'API
- python3 init_predictions.py
(ce fichier python lance src/training.py)

# Tests
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


# Introduction: 

Dans le cadre de ce projet MLOps, nous avons travaillé à l’industrialisation complète d’un modèle de prédiction météo basé sur le dataset WeatherAUS. L’objectif était de passer d’un modèle de machine learning entraîné localement à une plateforme plus structurée, capable de gérer l’ensemble du cycle de vie du modèle : préparation et versioning des données, entraînement, suivi des expériences, exposition via API, monitoring et automatisation du pipeline.

Le projet s’appuie sur plusieurs briques complémentaires : DVC et PostgreSQL pour la gestion des données, MLflow pour le suivi des expériences et la gestion des modèles, FastAPI pour exposer le modèle, Prometheus et Grafana pour le monitoring, Streamlit pour l’interface de démonstration, Docker Compose pour rendre l’environnement reproductible, GitHub Actions pour l’intégration continue et Airflow pour orchestrer les différentes étapes du pipeline.

L’enjeu principal n’était donc pas uniquement la performance du modèle, mais la mise en place d’une architecture MLOps cohérente, reproductible et observable, proche d’un fonctionnement industrialisé.


# Dataset WeatherAUS et objectifs MLOps : 

Le dataset utilisé est WeatherAUS, un jeu de données météorologiques australien. Il contient des observations historiques issues de plusieurs stations météo, avec des variables comme la température, l’humidité, la pression atmosphérique, la vitesse du vent ou encore les précipitations.

La variable cible est RainTomorrow. C’est une variable binaire qui indique s’il a plu le lendemain ou non.
D’un point de vue machine learning, on est donc sur un problème de classification binaire. L’objectif est de prédire la classe positive, c’est-à-dire la pluie du lendemain. 


# DVC et PostgreSQL : 
 
La gestion des données du projet repose sur une logique de traçabilité et de centralisation. Le dataset WeatherAUS est versionné avec DVC afin de suivre les différentes versions des données utilisées pour l’entraînement, sans stocker directement les fichiers volumineux dans Git. Git conserve ainsi le code source, tandis que DVC permet de rattacher une version précise du dataset à une version donnée du projet.
 
Les données sont ensuite chargées dans une base PostgreSQL via un script dédié. L’objectif est de ne plus dépendre uniquement d’un fichier CSV local, mais de disposer d’un stockage structuré et persistant. PostgreSQL permet de centraliser les données et de les rendre accessibles aux différents composants du pipeline, notamment les scripts d’entraînement et les services exécutés dans Docker.
 
L’association de DVC et PostgreSQL répond donc à deux enjeux complémentaires : DVC assure le versioning et la traçabilité des données, tandis que PostgreSQL assure leur stockage structuré et leur exploitation par la plateforme MLOps. Cette organisation améliore la reproductibilité du projet et rapproche l’architecture d’un fonctionnement plus industrialisé.

# MLflow & Registry 

Plateforme qui enregistre les entraînements de modèles. Un entraînement équivaut à 1 Run, et une sauvegarde des hyper paramètres, des métriques, des statuts Production et Archivés. Dans notre projet, seule la métrique F1-score est utilisée. 
Les enregistrements se situent dans Model Regitry. Ils permettent de suivre les expériences, et surtout, les modèles !  Afin de les comparer et de sélectionner le meilleur
L'avantage aussi est de revenir à une version antérieure en cas d'échec. Ce qui évite une manualisation complexe des modèles, l'absence d'historique, des expériences et des retours en arrière hasardeux.


# API, Authentification basique et OAuth2, tests Unitaires

Nous avons établi plusieurs routes pour notre API, autrement dit, des endpoints, pour l'entraînement, la prédiction et une route spécifique : 'webhook/grafana'. Ces routes sont des protocoles HTTP avec les méthodes GET et POST. La première pour récupérer des données, la seconde pour les traiter.

Concernant l'authentification, dans le projet, nous pouvions sécuriser avec une authentification basique ou une 'open autorisation' (OAuth.2) avec les tokens, nous avons réalisé les deux.
Pour la continuité du projet, nous sommes restés sur une authentification basique, car, notre objectif consistait à comprendre, apprendre le concept Devops pour les modèles de machine learning.
Pour écourter ce sujet, on a utilisé des variables d'environnements pour l'authentification basique, et le format JSON pour les tokens.

Pytest, une bibliothèque très prisée en programmation python, évidement utilisée dans notre projet et pour les tests unitaires. Nous avons donc utilisé des assertions qui vérifient si une condition est vraie. Des mocks pour créer des objets factices (api, bd). Nous avons établi des tests unitaires (qui vérifient le fonctionnement par portion de code), et non des tests d'intégration (plus ciblés sur le relationnel entre les composants).
Des fichiers ont été créés spécifiquement pour le chargement des données, la prédiction, l'entrainement et le fichier principal. Le préfixe 'test_' est très important car il permet à la bibliothèque Pytest de reconnaitre le fichier les tests.




# Prometheus & Grafana

Prometheus est un vrai collectionneur de métriques. Dans le projet, il recueille les métriques ressources système qui, à priori, sont les bases pour une surveillance optimale d'un système. 
Et Grafana, une extension de Prometheus, projette ces métriques-là dans une visualisation.
Lorsque l'on utilise ces plateformes, on doit adapter des métriques métiers au projet. C’est-à-dire bien les choisir afin de répondre à des besoins précis. Voyons l'exemple de webhook/grafana.
La route /webhook/grafana est un point de contact à créer, via la plateforme grafana. Dans notre cas, son utilité est d'envoyer une requête http, afin de déclencher un ré-entraînement du modèle. Sous quelle condition ? Si une métrique dépasse un certain seuil, comme par exemple un taux d'erreur de prédiction. On peut définir le seuil nous-mêmes, en acceptant 10%, 15% ou 20%. Mais une fois celle-ci dépassée, cela provoque automatiquement le ré-entraînement via cet endpoint.


# Dockerisation :
 
La dockerisation a pour objectif de rendre l’environnement du projet reproductible et portable. Plutôt que d’installer manuellement chaque service sur une machine locale, l’ensemble de la plateforme est décrit dans un fichier Docker Compose. Cela permet de lancer les différents composants du projet avec une commande unique, notamment PostgreSQL, MLflow, FastAPI, Prometheus, Grafana et Airflow.

Chaque service fonctionne dans son propre conteneur, avec ses dépendances, ses ports, ses variables d’environnement et ses volumes persistants. Docker Compose crée également un réseau interne permettant aux services de communiquer entre eux par leur nom de service, par exemple entre l’API, PostgreSQL, MLflow et Airflow.

Cette approche facilite fortement la reprise du projet par un autre membre de l’équipe. En lançant la commande docker compose up -d --build, l’ensemble de l’environnement peut être reconstruit de manière cohérente. Docker constitue donc la couche d’infrastructure qui permet d’exécuter la plateforme MLOps de façon stable, reproductible et mieux maîtrisée.


# Ci – Github action :
 
La partie CI repose sur GitHub Actions, avec l’objectif de vérifier automatiquement que le projet reste fonctionnel à chaque modification du code. Le workflow se déclenche lors d’un push ou d’une pull request, puis exécute plusieurs étapes de contrôle avant d’intégrer les changements.

La première étape consiste à installer Python et les dépendances du projet à partir du fichier requirements.txt. Le workflow lance ensuite un contrôle de qualité du code avec Flake8, principalement ciblé sur les erreurs critiques de syntaxe ou d’import. Il exécute également les tests unitaires avec Pytest, afin de vérifier les principales briques du projet, notamment l’API, la prédiction, l’entraînement et le chargement des données.

Enfin, la pipeline vérifie que l’image Docker de l’API peut être construite correctement. Cette CI permet donc de détecter rapidement les régressions, d’éviter d’intégrer du code cassé et de sécuriser le travail collaboratif. Elle ne constitue pas encore un déploiement continu complet, mais elle apporte une vraie base d’intégration continue.


# AIrflow :
 
Airflow est utilisé pour orchestrer automatiquement le pipeline MLOps. Avant son intégration, les différentes étapes du projet devaient être lancées manuellement, dans un ordre précis. L’objectif était donc de rendre cet enchaînement plus fiable, observable et relançable.

Le DAG principal mis en place s’appelle weather_training_pipeline. Il exécute trois tâches successives : le chargement des données dans PostgreSQL, l’entraînement du modèle avec enregistrement des résultats dans MLflow, puis la vérification du bon fonctionnement de l’API FastAPI. Cette organisation permet de formaliser clairement les dépendances entre les étapes du pipeline.

Airflow apporte également une interface de suivi permettant de visualiser l’état des tâches, de consulter les logs et de relancer une étape en cas d’échec. Son intégration a nécessité une base PostgreSQL dédiée afin d’éviter les conflits avec MLflow, ainsi qu’un ajustement des volumes et des permissions pour permettre l’écriture des artefacts MLflow. Airflow apporte donc une logique d’automatisation et de pilotage du pipeline de bout en bout.










# Interface Streamlit 

Overview : Aperçu des rôles des pages
Prédiction : Manipuler les variables d'entrées du dataset, via des sliders, afin de prédire la présence de Pluie ou non (à travers un bouton)
MLflow & Registry : Enregistrements et aperçu des entrainements du modèle, comparaison et sélection du meilleur modèle
Monitoring Prometheus & Grafana : Visualisation des métriques ressources systèmes afin de veiller au déroulement du processus API, ainsi que des métriques métiers du projet.
API Status : Vérifier l'état foncionnels des composants du projet

Overview	Prédiction
	

MLflow & Registry	Monitoring
	









API Status




# Conclusion

L'objectif du MLOPS est de construire un pipeline automatisé, déployé, reproductible, traçable, ré-entrainable et monitoré.

La limite du projet a été un dataset statique, provenant d'un projet précédent ainsi qu'un modèle exclusif. Un bruit gaussien a été ajouté afin de nuancer les variables numériques, correspondants aux mesures météorologiques, lors des entrainements.

En terme de perpectives, on pourrait déployer l'API avec Kuberntes, afin de gérer les montées en charge comme le passe de 100 requêtes à 10 000 requêtes (auto-scaling, load balacing).

Nous avons volontairement préféré une architecture simple mais fonctionnelle afin de mettre en pratique l'ensemble des modules appris et du cycle de vie d'un projet MLOp.
