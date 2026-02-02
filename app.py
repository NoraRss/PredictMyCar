import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import random
import numpy as np


st.set_page_config(page_title="Prix voiture", layout="wide")

@st.cache_data
def load_data(path="data/autoscrap_FIN_clean.csv"):
    """Charger le CSV nettoyé et le mettre en cache."""
    return pd.read_csv(path, sep=";")

@st.cache_resource
def load_model(path="ML/best_model.joblib"):
    """Charger le CSV nettoyé et le mettre en cache."""
    bundle = joblib.load(path)
    return (
        bundle["model"],
        bundle["features"],
        bundle["num_cols"],
        bundle["cat_cols"]
    )

df = load_data()
model, features, num_cols, cat_cols = load_model()



def prepare_input(df_train, user_input, features, cat_cols):
    """Préparer une ligne d'entrée pour la prédiction avec encodage des variables catégorielles."""
    input_df = pd.DataFrame([user_input])
    combined = pd.concat([df_train, input_df], axis=0)
    encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    for col in features:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded[features].tail(1)

def filter_df_by_inputs(df, user_input, num_cols, cat_cols):
    """Filtrer le DataFrame selon les critères numériques et catégoriels fournis."""
    filtered_df = df.copy()
    for col in num_cols:
        lo, hi = user_input[col]
        filtered_df = filtered_df[
            (filtered_df[col] >= lo) & (filtered_df[col] <= hi)
        ]
    for col in cat_cols:
        if user_input[col] is not None:
            filtered_df = filtered_df[filtered_df[col] == user_input[col]]
    return filtered_df

def generate_inputs(df, user_input, num_cols, cat_cols, step_dict=None, max_combos=300):
    """Générer plusieurs combinaisons d'inputs plausibles pour la prédiction."""
    filtered_df = filter_df_by_inputs(df, user_input, num_cols, cat_cols)
    if filtered_df.empty:
        filtered_df = df

    samples = []
    for _ in range(max_combos):
        sample = {}
        for col in num_cols:
            lo, hi = user_input[col]
            if lo == hi:
                sample[col] = lo
            else:
                step = step_dict.get(col, 1) if step_dict else 1
                sample[col] = random.choice(list(range(lo, hi + 1, step)))
        for col in cat_cols:
            sample[col] = (
                user_input[col]
                if user_input[col] is not None
                else random.choice(filtered_df[col].unique())
            )
        samples.append(sample)
    return samples

def is_reset_state(user_input, defaults):
    """Vérifier si tous les inputs sont dans l'état par défaut."""
    return all(user_input[k] == defaults[k] for k in defaults)

def build_quantile_inputs(df, num_cols, cat_cols, qs=(0.1, 0.5, 0.9)):
    """Créer des inputs basés sur les quantiles des variables numériques."""
    inputs = []
    for q in qs:
        inp = {}
        for col in num_cols:
            inp[col] = int(df[col].quantile(q))
        inp["annee"] = int(df["annee"].quantile(q))  
        for col in cat_cols:
            inp[col] = None
        inputs.append(inp)
    return inputs

# =========================================================
# Tabs
# =========================================================
tabs = st.tabs(["🏠 Accueil", "📊 Données", "📈 Visualisation", "💰 Prédiction", "🗺️ Cartographie"])


# =========================
# ACCUEIL
# =========================
with tabs[0]:
    st.title("🚗 Bienvenue sur PredictMyCar")

    st.markdown("""
PredictMyCar est votre assistant pour estimer le prix des voitures d’occasion en France en se basant sur les données historiques du marché. 
L’application vous fournit une estimation réaliste du prix pour un profil de voiture donné et vous montre la variabilité des prix observée pour des véhicules similaires.

---

🔍 **Que fait cette application ?**

                
1️⃣ **Prédiction du prix d'une voiture**  
                
Renseignez les critères qui vous intéressent (kilométrage, année, puissance, marque, modèle, carburant, boîte de vitesse, etc.) et l’application calcule :  
• 💰 Un prix moyen estimé  
• 📉📈 Un intervalle de prix probable, reflétant la variabilité du marché  
Plus les critères sont précis, plus l’estimation est affinée et représentative du profil recherché.
Si vous laissez certains critères non renseignés, l’application utilisera les valeurs médianes du marché pour vous proposer une estimation réaliste.

---

2️⃣ **Exploration des données**  

• Parcourez les données réelles issues d’annonces françaises de voitures d’occasion provenant du site Autosphère.  
• Indiquez vos critères de recherche et découvrez instantanément le nombre d’annonces correspondantes, leur kilométrage moyen, leur prix moyen, ainsi que les annonces détaillées.  
• 📊 Visualisez la répartition des véhicules selon différentes variables : kilométrage, année, puissance, marque, carburant, boîte de vitesse, etc.  
• Comparez le prix moyen selon chaque critère et identifiez les facteurs qui influencent la valeur d’une voiture.  

---

3️⃣ **Cartographie interactive**  

• 🗺️ Découvrez la répartition géographique des prix moyens par région.  
• ⚙️ Filtrez par marque, modèle, kilométrage, année, puissance, carburant ou boîte de vitesse.  
• Observez en un coup d’œil quelles régions offrent les prix les plus attractifs et où se trouvent les véhicules correspondant à vos critères.  

---

⚙️ **Comment ça fonctionne ?**  

• PredictMyCar repose sur un modèle prédictif entraîné sur des milliers d’annonces de voitures d’occasion provenant du site Autosphère.  
• La prédiction s’adapte selon vos critères et propose un intervalle de prix pour refléter la variabilité du marché.  
• Les visualisations interactives facilitent la compréhension des données et vous aident à prendre des décisions éclairées pour acheter ou vendre.

Cette application a été développée exclusivement à des fins académiques.
""")




# =========================
# DONNÉES
# =========================
with tabs[1]:
    st.header("📊 Données véhicules")

    view_mode = st.radio("Mode d’affichage", ["📋 Tableau", "🚗 Annonces"], horizontal=True)

    search_text = st.text_input("🔍 Rechercher une annonce (marque, modèle, carburant...)", placeholder="Ex : Peugeot diesel automatique")
    df_search = df.copy()

    if search_text:
        search_words = search_text.lower().split()  
        def match_row(row):
           row_text = " ".join(row.astype(str)).lower()
           return all(word in row_text for word in search_words)
    
        df_search = df_search[df_search.apply(match_row, axis=1)]


    if not df_search.empty:
        col1, col2, col3 = st.columns(3)

        col1.markdown(
            f"<div style='padding:15px; border-radius:10px; background-color:#FFA500; color:white; text-align:center;'>"
            f"<h3>📄 {len(df_search)}</h3>"
            f"<p>Annonces</p></div>", unsafe_allow_html=True
        )

        col2.markdown(
            f"<div style='padding:15px; border-radius:10px; background-color:#63C76A; color:white; text-align:center;'>"
            f"<h3>💶 {df_search['prix'].mean():,.0f} €</h3>"
            f"<p>Prix moyen</p></div>", unsafe_allow_html=True
        )

        col3.markdown(
            f"<div style='padding:15px; border-radius:10px; background-color:#FF6B6B; color:white; text-align:center;'>"
            f"<h3>🚗 {df_search['kilometrage'].mean():,.0f} km</h3>"
            f"<p>Kilométrage moyen</p></div>", unsafe_allow_html=True
        )

        st.divider()


    if view_mode == "📋 Tableau":
        st.dataframe(df_search.head(20), use_container_width=True, hide_index=True)
    else:
        st.markdown("### 🚗 Annonces véhicules")
        df_display = df_search.head(15)

        cols_per_row = 3
        for i in range(0, len(df_display), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, row in enumerate(df_display.iloc[i:i+cols_per_row].itertuples()):
                col = cols[j]
                bg_color = "#2E2E2E" if st.get_option("theme.base")=="dark" else "#E9D7EA"
                text_color = "#FFFFFF" if st.get_option("theme.base")=="dark" else "#000000"
                col.markdown(f"""
                    <div style='border:1px solid #e0e0e0; border-radius:10px; padding:15px; margin-bottom:12px; background-color:{bg_color}; color:{text_color}; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);'>
                        <h4 style="margin-bottom:5px; color:#FFA500;">🚘 {row.marque} {row.modele} ({row.annee})</h4>
                        💰 {row.prix:,.0f} €<br>📏 {row.kilometrage:,.0f} km<br>⛽ {row.carburant}<br>⚙️ {row.boite_vitesse}<br>🔋 {row.puissance_reelle_ch} ch<br>
                        <a href='{getattr(row,'url_annonce','#')}' target='_blank' style='color:#FFA500;font-weight:bold;'>Voir l’annonce</a>
                    </div>
                """, unsafe_allow_html=True)




# =========================
# VISUALISATION
# =========================
with tabs[2]:
    st.subheader("🔍 Exploration des variables")

    numeric_vars = ["kilometrage", "annee", "puissance_fiscale_cv", "puissance_reelle_ch"]
    categorical_vars = ["carburant", "boite_vitesse", "marque"]
    all_vars = numeric_vars + categorical_vars

    selected_var = st.selectbox("Choisir une variable à explorer", all_vars)

    col1, col2 = st.columns(2)


# ----------------------------
# GRAPHIQUE 1 : Répartition
# ----------------------------
with col1: 
    if selected_var in numeric_vars:
        if selected_var == "kilometrage":
            max_km = int(df['kilometrage'].max())
            bins = list(range(0, max_km + 10000, 10000))  
            
            fig1 = px.histogram(
                df,
                x='kilometrage',
                nbins=len(bins)-1,
                title="Répartition du kilométrage"
            )
            fig1.update_traces(marker_line_color='black', marker_line_width=1)
            
            fig1.update_xaxes(
                tickvals=bins,
                ticktext=[f"{b//1000}k" for b in bins]
            )

        else:
            fig1 = px.histogram(
                df,
                x=selected_var,
                nbins=30,
                title=f"Répartition de {selected_var}"
            )
            fig1.update_traces(marker_line_color='black', marker_line_width=1)

    else:
        if selected_var == "annee":
            df['annee_str'] = df['annee'].astype(str)
            counts = df['annee_str'].value_counts().sort_index().reset_index()
            counts.columns = ['annee', 'count']
            counts["percent"] = counts["count"] / counts["count"].sum() * 100

            fig1 = px.bar(
                counts,
                x='annee',
                y="percent",
                hover_data={"count": True, "percent": ':.2f'},
                title=f"Répartition de {selected_var} (%)",
                labels={"percent": "%"}
            )
            fig1.update_traces(marker_line_color='black', marker_line_width=1)

            fig1.update_xaxes(type='category', categoryorder='category ascending')

            fig1.update_layout(
                xaxis_tickangle=-45,
                xaxis_tickfont=dict(size=10),
                margin=dict(b=150)
            )
        else:
            counts = df[selected_var].value_counts().reset_index()
            counts.columns = [selected_var, "count"]
            counts["percent"] = counts["count"] / counts["count"].sum() * 100

            fig1 = px.bar(
                counts,
                x=selected_var,
                y="percent",
                hover_data={"count": True, "percent": ':.2f'},
                title=f"Répartition de {selected_var} (%)",
                labels={"percent": "%"}
            )
            fig1.update_traces(marker_line_color='black', marker_line_width=1)
            fig1.update_layout(
                xaxis_tickangle=-45,
                xaxis_tickfont=dict(size=10),
                margin=dict(b=150)
            )

    st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# GRAPHIQUE 2 : Variable vs Prix
# ----------------------------
with col2:
    if selected_var in numeric_vars:
        if selected_var == "kilometrage":
            max_km = int(df['kilometrage'].max())
            bins = list(range(0, max_km + 10000, 10000))
            
            df['bin'] = pd.cut(df['kilometrage'], bins=bins, include_lowest=True)
            bin_categories = df['bin'].cat.categories  
            df['bin_str'] = df['bin'].astype(str)
            
            df_grouped = df.groupby('bin_str', observed=True)['prix'].mean().reindex([str(b) for b in bin_categories]).reset_index()
            
            fig2 = px.bar(
                df_grouped,
                x='bin_str',
                y='prix',
                title=f"Prix moyen selon {selected_var}",
                labels={'bin_str': 'Kilométrage', 'prix': 'Prix moyen'},
                color_discrete_sequence=["#FAAE63"]
            )
            fig2.update_traces(marker_line_color='black', marker_line_width=1)
            

            fig2.update_xaxes(
                tickvals=[str(interval) for interval in bin_categories],
                ticktext=[f"{int(interval.left/1000)}k-{int(interval.right/1000)}k" for interval in bin_categories],
                tickangle=-45
            )

        

        elif selected_var == "puissance_fiscale_cv":
            df_grouped = df.groupby(selected_var)['prix'].mean().reset_index()
            fig2 = px.bar(
                df_grouped,
                x=selected_var,
                y='prix',
                color_discrete_sequence=["#FAAE63"]
            )


        elif selected_var == "puissance_reelle_ch":
            min_ch = int(df['puissance_reelle_ch'].min())
            max_ch = int(df['puissance_reelle_ch'].max())
    
            start = (min_ch // 10) * 10
            end = ((max_ch // 10) + 1) * 10
            bins = list(range(start, end + 1, 10))
    
            df['bin'] = pd.cut(df['puissance_reelle_ch'], bins=bins, include_lowest=True, right=False )
            bin_categories = df['bin'].cat.categories
            df['bin_str'] = df['bin'].astype(str)
    
            df_grouped = df.groupby('bin_str', observed=True)['prix'].mean().reindex([str(b) for b in bin_categories]).reset_index()
    
            fig2 = px.bar(
               df_grouped,
               x='bin_str',
               y='prix',
               title=f"Prix moyen selon {selected_var}",
               labels={'bin_str': 'Puissance ch', 'prix': 'Prix moyen'},
               color_discrete_sequence=["#FAAE63"]
            )
            fig2.update_traces(marker_line_color='black', marker_line_width=1)
    
            fig2.update_xaxes(
                tickvals=[str(interval) for interval in bin_categories],
                ticktext=[f"{int(interval.left)}-{int(interval.right)-1}" for interval in bin_categories],
                tickangle=-45
            )

        elif selected_var == "annee":
            df_grouped = df.groupby('annee')['prix'].mean().reset_index()

            df_grouped = df_grouped.sort_values(by='annee')

            fig2 = px.bar(
               df_grouped,
               x='annee',
               y='prix',
               title=f"Prix moyen selon {selected_var}",
               labels={'prix': 'Prix moyen'},
               color_discrete_sequence=["#FAAE63"]
            )
            fig2.update_traces(marker_line_color='black', marker_line_width=1)
            fig2.update_layout(xaxis_tickangle=-45, margin=dict(b=100))


        else:
            bins = 10
            df['bin'] = pd.cut(df[selected_var], bins=bins, include_lowest=True)
            bin_categories = df['bin'].cat.categories
            df['bin_str'] = df['bin'].astype(str)
            df_grouped = df.groupby('bin_str', observed=True)['prix'].mean().reindex([str(b) for b in bin_categories]).reset_index()

            fig2 = px.bar(
                df_grouped,
                x='bin_str',
                y='prix',
                title=f"Prix moyen en fonction de {selected_var}",
                labels={'bin_str': selected_var, 'prix': 'Prix moyen'},
                color_discrete_sequence=["#FAAE63"]
            )
            fig2.update_traces(marker_line_color='black', marker_line_width=1)
            fig2.update_xaxes(tickangle=-45)

    else:
        df_grouped = df.groupby(selected_var)['prix'].mean().reset_index()
        df_grouped = df_grouped.sort_values(by='prix', ascending=False)  
        fig2 = px.bar(
            df_grouped,
            x=selected_var,
            y='prix',
            title=f"Prix moyen selon {selected_var}",
            labels={'prix': 'Prix moyen'},
            color_discrete_sequence=["#FAAE63"]
        )
        fig2.update_traces(marker_line_color='black', marker_line_width=1)
        fig2.update_layout(xaxis_tickangle=-45, margin=dict(b=150))

    st.plotly_chart(fig2, use_container_width=True)



# =========================
# PRÉDICTION
# =========================
with tabs[3]:
    st.header("💰 Estimation du prix")

    marque_to_modeles = df.groupby("marque")["modele"].unique().apply(sorted).to_dict()
    modele_to_marque = df.drop_duplicates("modele").set_index("modele")["marque"].to_dict()

    DEFAULTS = {
        "kilometrage": (int(df["kilometrage"].min()), int(df["kilometrage"].max())),
        "puissance_fiscale_cv": (int(df["puissance_fiscale_cv"].min()), int(df["puissance_fiscale_cv"].max())),
        "puissance_reelle_ch": (int(df["puissance_reelle_ch"].min()), int(df["puissance_reelle_ch"].max())),
        "annee": (int(df["annee"].min()), int(df["annee"].max())),
        "carburant": None,
        "boite_vitesse": None,
        "marque": None,
        "modele": None
    }

    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    col_title, col_reset = st.columns([3, 1])
    with col_title:
        st.subheader("Caractéristiques")
    with col_reset:
        if st.button("🔄 Réinitialiser les filtres"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v

    col_num, col_center = st.columns([3, 3])

    with col_num:
        st.slider("Kilométrage (km)",
                  int(df["kilometrage"].min()),
                  int(df["kilometrage"].max()),
                  step=1000,
                  key="kilometrage")
        st.slider("Année de sortie",
                  int(df["annee"].min()),
                  int(df["annee"].max()),
                  step=1,
                  key="annee")
        st.slider("Puissance fiscale (CV)",
                  int(df["puissance_fiscale_cv"].min()),
                  int(df["puissance_fiscale_cv"].max()),
                  step=1,
                  key="puissance_fiscale_cv")
        st.slider("Puissance dynamique",
                  int(df["puissance_reelle_ch"].min()),
                  int(df["puissance_reelle_ch"].max()),
                  step=5,
                  key="puissance_reelle_ch")
        

    with col_center:
        st.selectbox("Carburant", sorted(df["carburant"].unique()),
                     index=None, key="carburant")
        st.selectbox("Boîte de vitesse", sorted(df["boite_vitesse"].unique()),
                     index=None, key="boite_vitesse")
        if st.session_state["modele"] and not st.session_state["marque"]:
            st.session_state["marque"] = modele_to_marque.get(st.session_state["modele"])

        available_models = (
            marque_to_modeles[st.session_state["marque"]]
            if st.session_state["marque"]
            else sorted(df["modele"].unique())
        )

        st.selectbox("Marque", sorted(df["marque"].unique()),
                     index=(sorted(df["marque"].unique()).index(st.session_state["marque"])
                            if st.session_state["marque"] else None),
                     key="marque")

        st.selectbox("Modèle", available_models,
                     index=(available_models.index(st.session_state["modele"])
                            if st.session_state["modele"] in available_models else None),
                     key="modele")

    user_input = {k: st.session_state[k] for k in DEFAULTS}
    current_year = pd.Timestamp.now().year
    annee_min, annee_max = user_input["annee"]
    user_input["age_voiture"] = (current_year - annee_max, current_year - annee_min)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("💰 Prédire le prix")


    if predict_clicked:
        preds = []


        if is_reset_state(user_input, DEFAULTS):
            quantile_inputs = build_quantile_inputs(df, num_cols, cat_cols)

            for inp in quantile_inputs:
                if "annee" in inp:
                    inp["age_voiture"] = current_year - inp["annee"]

                X = pd.DataFrame([inp])

                for col in cat_cols:
                    if col not in X or X[col].isna().any():
                        X[col] = "Autre"

                preds.append(np.expm1(model.predict(X)[0]))

            p10, p50, p90 = preds

            st.success(
                f"💰 **Prix de référence (profil médian)**  \n"
                f"💰 Prix estimé : {p50:,.0f} €  \n"
                f"Intervalle probable : {p10:,.0f} € – {p90:,.0f} €"
            )

    
        else:
            step_sizes = {"kilometrage": 5000, "puissance_fiscale_cv": 1, "age_voiture": 1}

            all_inputs = generate_inputs(
                df, user_input, num_cols, cat_cols,
                step_dict=step_sizes
            )

            for inp in all_inputs:
                if "annee" in inp:
                    inp["age_voiture"] = current_year - inp["annee"]

                X = pd.DataFrame([inp])

                for col in cat_cols:
                    if col not in X or X[col].isna().any():
                        X[col] = "Autre"

                preds.append(np.expm1(model.predict(X)[0]))

            preds = pd.Series(preds)

            st.success(
                f"💰 Prix moyen : {preds.mean():,.0f} €  \n"
                f"Intervalle probable : {preds.quantile(0.1):,.0f} € – {preds.quantile(0.9):,.0f} €"
            )



def safe_slider(label, min_val, max_val, value=None, step=1, key=None):
    """Créer un slider Streamlit sûr même si min_val et max_val sont identiques."""
    if min_val == max_val:
        max_val = min_val + 1  
        if value is None:
            value = (min_val, min_val)
    elif value is None:
        value = (min_val, max_val)
    return st.slider(label, min_value=min_val, max_value=max_val, value=value, step=step, key=key)




# =========================
# CARTOGRAPHIE INTERACTIVE AVANCÉE
# =========================
with tabs[4]:
    st.header("🗺️ Cartographie des prix moyens par région")

    df_map = df.copy()

    st.subheader("Filtres avancés")
    col1, col2, col3 = st.columns(3)

    with col1:
        marques = st.multiselect("Marque", sorted(df_map["marque"].unique()))
        if marques:
            df_map = df_map[df_map["marque"].isin(marques)]

        available_models = (
            sorted(df_map[df_map["marque"].isin(marques)]["modele"].unique())
            if marques else sorted(df_map["modele"].unique())
        )
        modeles = st.multiselect("Modèle", available_models)
        if modeles:
            df_map = df_map[df_map["modele"].isin(modeles)]

    with col2:
        km_min, km_max = safe_slider(
            "Kilométrage (km)",
            int(df_map["kilometrage"].min()),
            int(df_map["kilometrage"].max()),
            step=1000,
            key="km_slider"
        )
        annee_min, annee_max = safe_slider(
            "Année",
            int(df_map["annee"].min()),
            int(df_map["annee"].max()),
            step=1,
            key="annee_slider"
        )
        puiss_min, puiss_max = safe_slider(
            "Puissance réelle (ch)",
            int(df_map["puissance_reelle_ch"].min()),
            int(df_map["puissance_reelle_ch"].max()),
            step=5,
            key="puissance_slider"
        )

    with col3:
        carburants = st.multiselect("Carburant", sorted(df_map["carburant"].unique()))
        if carburants:
            df_map = df_map[df_map["carburant"].isin(carburants)]

        boites = st.multiselect("Boîte de vitesse", sorted(df_map["boite_vitesse"].unique()))
        if boites:
            df_map = df_map[df_map["boite_vitesse"].isin(boites)]


    df_map = df_map[
        (df_map["kilometrage"] >= km_min) & (df_map["kilometrage"] <= km_max) &
        (df_map["annee"] >= annee_min) & (df_map["annee"] <= annee_max) &
        (df_map["puissance_reelle_ch"] >= puiss_min) & (df_map["puissance_reelle_ch"] <= puiss_max) 
    ]

    if df_map.empty:
        st.info("Aucune annonce à afficher avec ces filtres.")
    else:
        cp_to_region = {
            **dict.fromkeys([75, 77, 78, 91, 92, 93, 94, 95], "Île-de-France"),
            **dict.fromkeys([1, 3, 7, 15, 26, 38, 42, 43, 63, 69, 73, 74], "Auvergne-Rhône-Alpes"),
            **dict.fromkeys([16, 17, 19, 23, 24, 33, 40, 47, 64, 79, 86, 87], "Nouvelle-Aquitaine"),
            **dict.fromkeys([22, 29, 35, 56], "Bretagne"),
            **dict.fromkeys([18, 28, 36, 37, 41, 45], "Centre-Val de Loire"),
            **dict.fromkeys([8, 10, 51, 52, 54, 55, 57, 67, 68, 88], "Grand Est"),
            **dict.fromkeys([2, 59, 60, 62, 80], "Hauts-de-France"),
            **dict.fromkeys([14, 27, 50, 61, 76], "Normandie"),
            **dict.fromkeys([21, 25, 39, 58, 70, 71, 89, 90], "Bourgogne-Franche-Comté"),
            **dict.fromkeys([44, 49, 53, 72, 85], "Pays de la Loire"),
            **dict.fromkeys([4, 5, 6, 13, 83, 84], "Provence-Alpes-Côte d'Azur"),
            **dict.fromkeys([9, 11, 12, 30, 31, 32, 34, 46, 48, 65, 66, 81, 82], "Occitanie"),
            **dict.fromkeys([2, 20], "Corse")
        }

        df_map["region"] = df_map["code_postal"].astype(str).str[:2].astype(int).map(cp_to_region)
        df_map = df_map.dropna(subset=["region"])  

        df_region = df_map.groupby("region")["prix"].mean().reset_index()
        df_region.columns = ["region", "prix_moyen"]

        import requests
        url_geojson = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions-version-simplifiee.geojson"
        regions_geojson = requests.get(url_geojson).json()

        df_region = df_map.groupby("region").agg(
            prix_moyen=("prix", "mean"),
            nb_annonces=("prix", "count")
        ).reset_index()


        df_region["hover_text"] = df_region.apply(
            lambda row: f"Nombre d'annonces : {row['nb_annonces']}<br>Prix moyen : {row['prix_moyen']:,.0f} €",
            axis=1
        )

        fig_map = px.choropleth_mapbox(
            df_region,
            geojson=regions_geojson,
            locations="region",
            featureidkey="properties.nom",
            color="prix_moyen",
            color_continuous_scale="YlOrRd",
            hover_name="region",          
            hover_data={"hover_text": False}, 
            custom_data=["hover_text"],       
            zoom=4.5,
            center={"lat":46.6,"lon":2.4},
            mapbox_style="carto-positron",
            height=650
        )
        fig_map.update_traces(marker_opacity=0.7)

        fig_map.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]}<extra></extra>"
        )

        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

