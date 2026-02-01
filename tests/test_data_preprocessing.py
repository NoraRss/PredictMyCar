import pandas as pd
import numpy as np
import pytest
from data_preprocessing import (
    add_features,
    process_nom_annonce,
    normalize_carburant,
    clean_data,
    load_data
)


def test_add_age_voiture_feature():
    """Vérifie que add_features crée la colonne 'age_voiture' avec les valeurs correctes."""
    df = pd.DataFrame({"annee": [2020, 2015, 2010]})
    df2 = add_features(df)

    assert "age_voiture" in df2.columns
    expected_ages = [6, 11, 16]  
    assert df2["age_voiture"].tolist() == expected_ages



def test_process_nom_annonce_splits_correctly():
    """Vérifie que process_nom_annonce sépare 'nom_annonce' en 'marque' et 'modele'."""
    df = pd.DataFrame({"nom_annonce": ["Renault Clio", "Peugeot 208", "Tesla Model3"]})
    df2 = process_nom_annonce(df)

    assert "marque" in df2.columns
    assert "modele" in df2.columns
    assert df2["marque"].tolist() == ["Renault", "Peugeot", "Tesla"]
    assert df2["modele"].tolist() == ["Clio", "208", "Model3"]
    assert "nom_annonce" not in df2.columns



def test_normalize_carburant_lowercases_and_standardizes():
    """Vérifie que normalize_carburant met tout en minuscules et standardise 'électrique'."""
    df = pd.DataFrame({"carburant": ["Diesel", "Electrique", "Essence", "ELECTRIQUE"]})
    df2 = normalize_carburant(df)

    expected_values = ["diesel", "électrique", "essence", "électrique"]
    assert df2["carburant"].tolist() == expected_values



def test_clean_data_removes_outliers_and_converts_code_postal():
    """Vérifie que clean_data supprime les outliers et convertit code_postal en int."""
    df = pd.DataFrame({
        "prix": [10000, 12000, 11000, 11500, 1000000],        
        "kilometrage": [50000, 51000, 52000, 53000, 700000],  
        "puissance_fiscale_cv": [4, 4, 5, 5, 50],            
        "puissance_reelle_ch": [100, 105, 110, 120, 1000],    
        "code_postal": ["75001", "69001", "13001", "75002", "69002"]
    })
    
    df_clean = clean_data(df)
    
   
    assert df_clean["prix"].max() < 1000000
    assert df_clean["kilometrage"].max() < 700000
    assert df_clean["puissance_fiscale_cv"].max() < 50
    assert df_clean["puissance_reelle_ch"].max() < 1000
    
    assert df_clean["code_postal"].dtype == int



def test_load_data_creates_dataframe(tmp_path):
    """Vérifie que load_data crée un DataFrame correct à partir d'un JSON."""
    data = [{"prix": 10000, "kilometrage": 50000}, {"prix": 12000, "kilometrage": 60000}]
    json_file = tmp_path / "test.json"
    import json
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    
    df = load_data(str(json_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert set(df.columns) == {"prix", "kilometrage"}
