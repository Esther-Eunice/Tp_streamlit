import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Charger les données Iris
data = load_iris()
iris_df = pd.DataFrame(
    data.data, columns=data.feature_names
)
iris_df['species'] = data.target
iris_df['species'] = iris_df['species'].map({
    0: 'setosa', 1: 'versicolor', 2: 'virginica'
})

# Titre de l'application
st.title("Exploration du jeu de données Iris")
st.sidebar.title("Options")

# Section d'affichage des données
st.header("Aperçu des données")
if st.checkbox("Afficher les données brutes"):
    st.dataframe(iris_df)

# Filtrer les données par espèce
species_filter = st.sidebar.multiselect(
    "Filtrer par espèce :", options=iris_df['species'].unique(), default=iris_df['species'].unique()
)
filtered_data = iris_df[iris_df['species'].isin(species_filter)]

# Résumé statistique
st.header("Résumé statistique")
st.write(filtered_data.describe())

# Visualisation des données
st.header("Visualisation des données")
plot_type = st.sidebar.selectbox(
    "Choisissez un type de graphique :", ["Scatterplot", "Histogramme"]
)

if plot_type == "Scatterplot":
    x_axis = st.sidebar.selectbox("Axe X :", options=data.feature_names, index=0)
    y_axis = st.sidebar.selectbox("Axe Y :", options=data.feature_names, index=1)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=filtered_data, x=x_axis, y=y_axis, hue='species', palette='Set1'
    )
    plt.title("Scatterplot")
    st.pyplot(plt)

elif plot_type == "Histogramme":
    feature = st.sidebar.selectbox("Choisissez une caractéristique :", options=data.feature_names)
    plt.figure(figsize=(10, 6))
    
    # Vérification du type de données
    if pd.api.types.is_numeric_dtype(filtered_data[feature]):
        sns.histplot(filtered_data, x=feature, kde=True, hue='species', palette='Set2')
        plt.title(f"Histogramme de {feature}")
    else:
        st.error(f"La colonne {feature} n'est pas de type numérique.")
        
    st.pyplot(plt)
