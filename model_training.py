from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "autoscrap_FIN_clean.csv"
ML_DIR = BASE_DIR / "ML"  
ML_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH, sep=";")
df.columns = df.columns.str.strip().str.lower()

target = "prix"
df["age_voiture"] = pd.Timestamp.now().year - df["annee"]


y = np.log1p(df[target].astype(float))

num_cols = ["kilometrage", "puissance_fiscale_cv", "puissance_reelle_ch", "age_voiture"]
cat_cols = ["carburant", "boite_vitesse", "marque", "modele"]

num_cols = [c for c in num_cols if c in df.columns]
cat_cols = [c for c in cat_cols if c in df.columns]

X = df[num_cols + cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=60
)

def encode_rare_categories(X_train, X_test, cat_cols, min_count=10):
    """
    Remplace les catégories rares par 'Autre' dans les colonnes catégorielles.
    
    Paramètres :
    - X_train, X_test : DataFrames d'entraînement et de test
    - cat_cols : colonnes catégorielles
    - min_count : seuil minimal pour conserver une catégorie

    Retour :
    - X_train et X_test transformés
    - dictionnaire des catégories connues après encodage
    """
     
    X_train = X_train.copy()
    X_test = X_test.copy()
    known_categories = {}

    for col in cat_cols:
        counts = X_train[col].value_counts()
        rare = counts[counts < min_count].index

        X_train[col] = X_train[col].replace(rare, "Autre")
        X_test[col] = X_test[col].apply(
            lambda x: x if x in X_train[col].unique() else "Autre"
        )

        known_categories[col] = X_train[col].unique().tolist()

    return X_train, X_test, known_categories

X_train, X_test, known_categories = encode_rare_categories(
    X_train, X_test, cat_cols
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(
        categories=[known_categories[c] for c in cat_cols],
        drop="first",
        handle_unknown="ignore"
    ), cat_cols)
])

models = {
    "DummyRegressor": DummyRegressor(),
    "LinearRegression": LinearRegression(),
    "LassoRegression": Lasso(random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
}


param_grids = {
"DummyRegressor": {},

"LinearRegression": {
    "model__fit_intercept": [True, False]
},

"LassoRegression": {
    "model__alpha": [0.001, 0.01, 0.1, 1, 10],
    "model__max_iter": [5000]
},

"KNeighborsRegressor": {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2]  
},

"DecisionTreeRegressor": {
    "model__max_depth": [3, 5, 10, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 5]
},

"RandomForestRegressor": {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
},

"GradientBoostingRegressor": {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5],
    "model__subsample": [0.8, 1.0]
},
}


best_pipeline = None
best_cv_mae = np.inf
best_model_name = None
results_list = []

for name, model in models.items():
    print(f"\n=== Training {name} ===")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    if param_grids[name]:
        grid = GridSearchCV(
            pipeline,
            param_grids[name],
            scoring="neg_mean_absolute_error",
            cv=5,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        cv_mae = -grid.best_score_
        best_estimator = grid.best_estimator_

    else:
        pipeline.fit(X_train, y_train)
        cv_mae = -cross_val_score(
            pipeline, X_train, y_train,
            scoring="neg_mean_absolute_error",
            cv=5
        ).mean()
        best_estimator = pipeline

    results_list.append({
        "model": name,
        "CV_MAE": cv_mae
    })

    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_pipeline = best_estimator
        best_model_name = name


y_pred_log = best_pipeline.predict(X_test)

y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae = mean_absolute_error(y_test_real, y_pred_real)
r2 = r2_score(y_test_real, y_pred_real)

print("\n=== Évaluation finale sur le jeu de test ===")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.3f}")


pd.DataFrame(results_list).to_csv(
    ML_DIR / "cv_models_metrics.csv",
    index=False
)

bundle = {
    "model": best_pipeline,
    "model_name": best_model_name,
    "cv_mae": best_cv_mae,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "features": num_cols + cat_cols, 
    "metrics": {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "target_log": True  
    },
}


joblib.dump(bundle, ML_DIR / "best_model.joblib")

print(f"\n🏆 Best model : {best_model_name}")
print("Modèle sauvegardé avec succès ✅")
