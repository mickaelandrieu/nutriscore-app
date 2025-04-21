"""Basic web app to explore the performance of the ML app."""
import joblib
import pandas as pd
import streamlit as st
from streamlit_extras.colored_header import colored_header

import config

# Configuration de la page
st.set_page_config(
    page_title="Nutriscore Predictor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions utilitaires
def load_categories():
    """Load the list of categories.

    Returns:
        a DataFrame.
    """
    return pd.read_csv(f"{config.DOCS}categories.csv")

def load_model():
    """Load Machine Learning model.

    Returns:
        an instance of sklearn classifier.
    """
    return joblib.load(f"{config.MODELS}best_model.z")

def load_features():
    """Load the required features for predictions.

    Returns:
        a DataFrame
    """
    return pd.read_csv(f"{config.DOCS}best_features.csv")

def load_samples():
    """Load the samples examples.

    Returns:
        a string content as markdown
    """
    with open(f"{config.DOCS}samples.md") as samples:
        return samples.read()

# Chargement des données
@st.cache_data
def load_data():
    model = load_model()
    categories = load_categories()
    features = load_features()["feature"]
    return model, categories, features

# Initialisation
model, categories, features = load_data()

# Interface utilisateur
st.title("🥗 Nutriscore Predictor")
st.markdown("""
    Cette application prédit le Nutriscore d'un produit alimentaire en fonction de ses caractéristiques nutritionnelles.
""")

# Sidebar avec les informations
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
        Entrez les informations nutritionnelles du produit pour obtenir sa prédiction de Nutriscore.
    """)
    
    with st.expander("Exemples de produits"):
        st.markdown(load_samples())

# Formulaire de saisie
colored_header(
    label="📊 Informations nutritionnelles",
    description="Entrez les valeurs nutritionnelles du produit",
    color_name="blue-70"
)

# Organisation en colonnes pour une meilleure présentation
col1, col2 = st.columns(2)

with col1:
    energy = st.number_input(
        "Énergie (kcal/100g)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="Énergie en kilocalories pour 100g de produit"
    )
    
    saturated_fat = st.number_input(
        "Graisses saturées (g/100g)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="Graisses saturées en grammes pour 100g de produit"
    )
    
    sugars = st.number_input(
        "Sucres (g/100g)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="Sucres en grammes pour 100g de produit"
    )

with col2:
    salt = st.number_input(
        "Sel (g/100g)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="Sel en grammes pour 100g de produit"
    )
    
    fibers = st.number_input(
        "Fibres (g/100g)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        help="Fibres en grammes pour 100g de produit"
    )
    
    category = st.selectbox(
        "Catégorie du produit",
        options=categories,
        help="Sélectionnez la catégorie du produit"
    )

# Préparation des données pour la prédiction
product = {
    "energy": round(float(energy)),
    "saturated_fat": round(float(saturated_fat)),
    "sugars": round(float(sugars)),
    "salt": round(float(salt)),
    "fibers": round(float(fibers)),
    "group1_Beverages": 0,
    "group1_Cereals and potatoes": 0,
    "group1_Composite foods": 0,
    "group1_Fat and sauces": 0,
    "group1_Fruits and vegetables": 0,
    "group1_Milk and dairy products": 0,
    "group1_Sugary snacks": 0,
    "group1_unknown": 0,
}

# Mise à jour de la catégorie
formatted_category = f"group1_{category}"
if formatted_category in product:
    product[formatted_category] = 1

# Bouton de prédiction
if st.button("Prédire le Nutriscore", type="primary"):
    sample = list(product.values())
    nutrigrade = model.predict([sample])[0].upper()
    
    # Affichage du résultat avec style
    st.markdown("---")
    colored_header(
        label="Résultat de la prédiction",
        description="Le Nutriscore prédit pour ce produit est :",
        color_name="green-70"
    )
    
    # Affichage stylisé du résultat
    st.markdown(f"""
        <div style='text-align: center; font-size: 2em; font-weight: bold; color: #2ecc71;'>
            {nutrigrade}
        </div>
    """, unsafe_allow_html=True)
    
    # Informations supplémentaires
    st.markdown("""
        <div style='text-align: center; margin-top: 1em;'>
            <small>Note: Cette prédiction est basée sur un modèle de machine learning et doit être utilisée à titre indicatif.</small>
        </div>
    """, unsafe_allow_html=True)
