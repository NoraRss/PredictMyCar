# 🚗 PredictMyCar

## Introduction

Estimer le prix d’une voiture d’occasion peut être complexe : plusieurs facteurs influencent sa valeur, comme l’année, le modèle, la puissance ou le kilométrage. Pour obtenir une estimation fiable, il est utile de se baser sur les données historiques du marché.

PredictMyCar est une application développée en Python qui permet de générer cette estimation automatiquement. L’utilisateur saisit les caractéristiques d’un véhicule et le modèle prédit le prix médian observé dans les annonces historiques. De plus, l’application fournit un intervalle de prix pour refléter la variation observée entre véhicules comparables, donnant ainsi une idée de la plage de prix réaliste pour ce profil de voiture.

Cette approche se base sur l’analyse de milliers d’annonces réelles collectées sur Autosphère, permettant de proposer des estimations contextualisées et cohérentes avec le marché français des voitures d’occasion.


## Fonctionnalités clés
- Prédiction du prix des véhicules
L’utilisateur saisit les caractéristiques d’une voiture à l'aide de sliders et menus déroulants.
L’application fournit une estimation médiane du prix basée sur les données historiques.

- Intervalle de prix : la plage affichée est calculé à partir de plusieurs entrées simulées dérivées du dataset, représentant de manière réaliste la variation des prix pour ce type de voiture.

- Visualisation des tendances du marché
Histogrammes et cartes interactives pour analyser les prix selon région, marque, type de carburant, etc...

- Exploration des données
Possibilité de filtrer par marque, modèle, année, kilométrage, carburant ou puissance pour mieux comprendre le marché.


## Etapes du projet :
- Scraping de données réelles pour obtenir des informations à jour sur le marché de l’occasion.

Nous avons réalisé un scraping sur le site Autosphere, plusieurs fois, en récupérant les données page par page (35 pages à chaque fois, sachant qu’une page contient environ 23 annonces donc environ 800 annonces à chaque fois), ce qui nous a permis d’obtenir au total environ 5 600 annonces.

Nous avons scrapés pour chaque annonce les données suivantes : Marque, Modèle, Prix, Année, Kilométrage, Puissance fiscale (CV), Puissance réelle (ch), Carburant, Boîte de vitesse, Code postal

- Nettoyage et prétraitement des données afin d’avoir un jeu fiable et exploitable.
- Prédiction du prix des véhicules via machine learning et sélection du modèle le plus performant.

Cette étape repose sur l’entraînement et la comparaison de plusieurs modèles de régression, notamment le Dummy Regressor, la régression linéaire, la régression Lasso, l’algorithme des k-plus proches voisins (KNN), l’arbre de décision, la forêt aléatoire (Random Forest) et le Gradient Boosting.

L’évaluation des performances est réalisée à l’aide d’un prétraitement des données et d’une validation croisée, en s’appuyant sur des métriques adaptées telles que l’erreur absolue moyenne (MAE), la racine de l’erreur quadratique moyenne (RMSE) et le coefficient de détermination (R²). Le modèle final est sélectionné comme étant celui présentant la plus faible erreur absolue moyenne (MAE) en validation croisée sur le jeu d’entraînement. 

- Création d'une interface web interactive avec Streamlit pour rendre la prédiction accessible à tous.



## Mode d’emploi
Lancer l’application web : python -m streamlit run app.py


## Structure du projet
```
PredictMyCar/
├── data/                     # Dossier pour les jeux de données
├── ML/                       # Dossier pour les fichiers liés au Machine Learning
├── tests/                    # Dossier pour les tests unitaires
│ ├── test_autoscrap.py
│ ├── test_data_preprocessing.py
│ └── test_model_training.py
├── app.py                   # Application streamlit
├── autoscrap_800.py         # Scraping des données (script des 35 premières pages, on changeait le nbr de pages à scraper et le  nom du json en sortie pour les autres pages)
├── data_fusion.py           # Fusion des données
├── data_preprocessing.py    # Préparation et nettoyage des données
├── model_training.py        # Machine Learning
└── README.md
```                  


## Technologies utilisées
Logiciel : Python 3.13.9

Bibliothèques :
- Scraping : Playwright 
- Nettoyage et prétraitement : Pandas, NumPy, Regex
- Visualisation : Matplotlib, Seaborn, Plotly
- Machine Learning : Scikit-learn (Linear Regression, Random Forest, Gradient Boosting…)
- Application web : Streamlit


## Auteurs 
Rousseau Nora

Boudamous Lyna











