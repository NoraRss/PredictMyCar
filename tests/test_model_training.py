import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

def test_train_test_split_sizes():
    """Vérifie que train_test_split divise correctement les données selon test_size."""
    df = pd.DataFrame({
        "x1": range(100),
        "x2": range(100),
        "y": range(100)
    })

    X = df[["x1", "x2"]]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=60
    )

    assert len(X_train) == 80
    assert len(X_test) == 20

def test_no_data_leakage():
    """Vérifie qu'il n'y a pas de chevauchement d'indices entre train et test."""
    df = pd.DataFrame({
        "x": range(50),
        "y": range(50)
    })

    X_train, X_test, _, _ = train_test_split(
        df[["x"]], df["y"], test_size=0.3, random_state=42
    )

    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_cross_validation_runs():
    """Vérifie que cross_val_score retourne le bon nombre de scores et que tous sont négatifs."""
    X = pd.DataFrame({
        "x1": range(40),
        "x2": range(40)
    })
    y = np.log1p(range(40))

    model = LinearRegression()
    scores = cross_val_score(
        model, X, y,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    assert len(scores) == 5
    assert np.all(scores < 0)


def test_model_training_runs():
    """Vérifie que le modèle peut s'entraîner et produire des prédictions de la bonne taille."""
    X = pd.DataFrame({
        "x1": range(30),
        "x2": range(30)
    })
    y = range(30)

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(X)
