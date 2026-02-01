# PredictMyCar

## Introduction

Estimer le prix d’une voiture d’occasion peut être complexe : plusieurs facteurs influencent sa valeur, comme l’année, le modèle, la puissance ou le kilométrage. Pour obtenir une estimation fiable, il est utile de se baser sur les données historiques du marché.

PredictMyCar est une application développée en Python qui permet de générer cette estimation automatiquement. L’utilisateur saisit les caractéristiques d’un véhicule et le modèle prédit le prix médian observé dans les annonces historiques. De plus, l’application fournit un intervalle de prix pour refléter la variation observée entre véhicules comparables, donnant ainsi une idée de la plage de prix réaliste pour ce profil de voiture.

Cette approche se base sur l’analyse de milliers d’annonces réelles collectées sur Autosphère, permettant de proposer des estimations contextualisées et cohérentes avec le marché français des voitures d’occasion.



PredictMyCar est un projet qui combine :
- Scraping de données réelles pour obtenir des informations à jour sur le marché de l’occasion.
- Nettoyage et prétraitement des données afin d’avoir un jeu fiable et exploitable.
- Prédiction du prix des véhicules via machine learning et sélection du modèle le plus performant.
- Création d'une interface web interactive avec Streamlit pour rendre la prédiction accessible à tous.


## Fonctionnalités clés
- Prédiction du prix des véhicules
L’utilisateur saisit les caractéristiques d’une voiture à l'aide de sliders et menus déroulants.
L’application fournit une estimation médiane du prix basée sur les données historiques.

- Intervalle de prix : la plage affichée est calculé à partir de plusieurs entrées simulées dérivées du dataset, représentant de manière réaliste la variation des prix pour ce type de voiture.

- Visualisation des tendances du marché
Histogrammes et cartes interactives pour analyser les prix selon région, marque, type de carburant, etc...

- Exploration des données
Possibilité de filtrer par marque, modèle, année, kilométrage, carburant ou puissance pour mieux comprendre le marché.



## Mode d’emploi
Lancer l’application web : python -m streamlit run app.py


## Structure du projet
PredictMyCar/
├── data/                       # Dossier pour les datasets
├── ML/                         # Dossier pour les fichiers liés au Machine Learning
├── tests/                      # Dossier pour les tests unitaires
│   ├── test_autoscrap.py       
│   ├── test_data_preprocessing.py 
│   └── test_model_training.py  
├── app.py                      # Application streamlit
├── autoscrap_800.py            # Scraping des données
├── data_fusion.py              # Fusion des données
├── data_preprocessing.py       # Préparation et nettoyage des données
├── model_training.py           # Machine Learning
└── README.md                   


## Technologies utilisées
Logiciel : Python
Bibliothèques :
- Scraping : Playwright 
- Nettoyage et prétraitement : Pandas, NumPy, Regex
- Visualisation : Matplotlib, Seaborn, Plotly
- Machine Learning : Scikit-learn (Linear Regression, Random Forest, Gradient Boosting…)
- Application web : Streamlit


# Auteurs 
Rousseau Nora
Boudamous Lyna
