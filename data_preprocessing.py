import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted") 


def load_data(json_path: str) -> pd.DataFrame:
    """Charger les données depuis un fichier JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajouter des colonnes dérivées comme l'âge de la voiture."""
    df = df.copy()
    if 'annee' in df.columns:
        df['age_voiture'] = 2026 - df['annee']
    return df


def process_nom_annonce(df: pd.DataFrame) -> pd.DataFrame:
    """
    Créer deux nouvelles colonnes à partir de 'nom_annonce':
    - 'marque': première partie
    - 'modele': reste du nom
    """
    df = df.copy()
    if 'nom_annonce' in df.columns:
        df[['marque', 'modele']] = df['nom_annonce'].str.split(pat=' ', n=1, expand=True)
        df = df.drop(columns=['nom_annonce'])
    return df


def normalize_carburant(df: pd.DataFrame) -> pd.DataFrame:
    """Standardiser la colonne 'carburant'."""
    df = df.copy()
    if 'carburant' in df.columns:
        df['carburant'] = df['carburant'].str.lower().replace({'electrique': 'électrique'})
    return df


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Identifier les outliers selon la méthode IQR."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series[(series < lower) | (series > upper)]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyer les données: retirer outliers et convertir les types."""
    df = df.copy()
    
    outliers_prix = detect_outliers_iqr(df['prix']) if 'prix' in df.columns else pd.Series(dtype=float)
    outliers_kilometrage = detect_outliers_iqr(df['kilometrage']) if 'kilometrage' in df.columns else pd.Series(dtype=float)
    
    outliers_puissance_fiscale = detect_outliers_iqr(df['puissance_fiscale_cv']) if 'puissance_fiscale_cv' in df.columns else pd.Series(dtype=float)
    outliers_puissance_reelle = detect_outliers_iqr(df['puissance_reelle_ch']) if 'puissance_reelle_ch' in df.columns else pd.Series(dtype=float)

    df_clean = df.copy()
    
    if 'prix' in df_clean.columns:
        df_clean = df_clean[~df_clean['prix'].isin(outliers_prix)]
    if 'kilometrage' in df_clean.columns:
        df_clean = df_clean[~df_clean['kilometrage'].isin(outliers_kilometrage)]
    if 'puissance_fiscale_cv' in df_clean.columns:
        df_clean = df_clean[~df_clean['puissance_fiscale_cv'].isin(outliers_puissance_fiscale)]
    if 'puissance_reelle_ch' in df_clean.columns:
        df_clean = df_clean[~df_clean['puissance_reelle_ch'].isin(outliers_puissance_reelle)]
    
    if 'code_postal' in df_clean.columns:
        df_clean['code_postal'] = df_clean['code_postal'].astype(int)
    
    return df_clean


def save_csv(df: pd.DataFrame, path: str):
    """Exporter le DataFrame nettoyé en CSV."""
    df.to_csv(path, index=False, encoding="utf-8-sig", sep=";")
    print(f"✅ Le fichier CSV a été créé : {path}")


def plot_eda(df: pd.DataFrame):
    """Visualisation: boxplots et histogrammes pour les variables clés."""
    plt.figure(figsize=(20,5))
    
    plt.subplot(1,4,1)
    if 'prix' in df.columns:
        sns.boxplot(y=df['prix'])
        plt.title('Boxplot Prix (€)')
    plt.subplot(1,4,2)
    if 'kilometrage' in df.columns:
        sns.boxplot(y=df['kilometrage'])
        plt.title('Boxplot Kilométrage (km)')
    plt.subplot(1,4,3)
    if 'puissance_fiscale_cv' in df.columns:
        sns.boxplot(y=df['puissance_fiscale_cv'])
        plt.title('Boxplot Puissance Fiscale (CV)')
    plt.subplot(1,4,4)
    if 'puissance_reelle_ch' in df.columns:
        sns.boxplot(y=df['puissance_reelle_ch'])
        plt.title('Boxplot Puissance Réelle (CH)')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(20,5))
    plt.subplot(1,4,1)
    if 'prix' in df.columns:
        sns.histplot(df['prix'], bins=30, kde=True)
        plt.title('Histogramme Prix (€)')
    plt.subplot(1,4,2)
    if 'kilometrage' in df.columns:
        sns.histplot(df['kilometrage'], bins=30, kde=True)
        plt.title('Histogramme Kilométrage (km)')
    plt.subplot(1,4,3)
    if 'puissance_fiscale_cv' in df.columns:
        sns.histplot(df['puissance_fiscale_cv'], bins=30, kde=True)
        plt.title('Histogramme Puissance Fiscale (CV)')
    plt.subplot(1,4,4)
    if 'puissance_reelle_ch' in df.columns:
        sns.histplot(df['puissance_reelle_ch'], bins=30, kde=True)
        plt.title('Histogramme Puissance Réelle (CH)')
    
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df: pd.DataFrame):
    """Afficher la corrélation entre variables numériques."""
    cols_corr = ['prix','annee','kilometrage','age_voiture']
    if 'puissance_fiscale_cv' in df.columns:
        cols_corr.append('puissance_fiscale_cv')
    if 'puissance_reelle_ch' in df.columns:
        cols_corr.append('puissance_reelle_ch')
    
    corr = df[cols_corr].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, mask=mask,
                square=True, linewidths=0.5, cbar_kws={"shrink":0.8})
    plt.title("Corrélation entre variables numériques", fontsize=16)
    plt.show()


if __name__ == "__main__":
    df = load_data("data/autoscrap_FIN.json")
    df = process_nom_annonce(df)
    df = add_features(df)
    df_clean = clean_data(df)
    df_clean = normalize_carburant(df_clean)

    desired_order = [
        'marque', 'modele', 'annee', 'age_voiture', 'carburant', 'boite_vitesse',
        'kilometrage', 'puissance_fiscale_cv', 'puissance_reelle_ch', 'prix', 'code_postal', 'url_annonce'
    ]
    df_clean = df_clean[[col for col in desired_order if col in df_clean.columns]]

    save_csv(df_clean, "data/autoscrap_FIN_clean.csv")

    print(df_clean.head())
    print(df_clean.columns)
    print(df_clean.info())
