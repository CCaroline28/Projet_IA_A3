#commande pour installer les bibliothèques : pip install pandas folium plotly scikit-learn 

import pandas as pd
import folium
from folium.plugins import HeatMap

# =====================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# =====================================================

# Importation du dataset nettoyé issu de la phase Big Data
df = pd.read_csv("export_IA.csv")

# Sélection des variables nécessaires pour la visualisation
colonnes = [
    "implantation_station",
    "puissance_nominale",
    "consolidated_longitude",
    "consolidated_latitude"
]

# Suppression des observations incomplètes
df = df[colonnes].dropna()

# Filtrage des coordonnées afin de conserver uniquement
# les bornes localisées en France métropolitaine
df = df[
    (df["consolidated_longitude"] >= -5.5) &
    (df["consolidated_longitude"] <= 9.5) &
    (df["consolidated_latitude"] >= 41) &
    (df["consolidated_latitude"] <= 51.5)
]

# =====================================================
# CRÉATION DES CATÉGORIES DE PUISSANCE
# =====================================================

# Classification des bornes selon leur puissance nominale
def categorie_puissance(p):
    if p <= 22:
        return "Normale ≤ 22 kW"
    elif p <= 50:
        return "Rapide 23-50 kW"
    elif p <= 150:
        return "Très rapide 51-150 kW"
    else:
        return "Ultra rapide > 150 kW"

# Ajout d'une variable catégorielle utilisée pour la visualisation
df["categorie_puissance"] = df["puissance_nominale"].apply(categorie_puissance)

# Création d'un échantillon afin d'améliorer les performances
# lors de l'affichage des cartes interactives
df_sample = df.sample(min(5000, len(df)), random_state=42)

# =====================================================
# FONCTION D'AJOUT DE TITRE AUX CARTES
# =====================================================

# Fonction permettant d'ajouter un titre visible sur une carte Folium
def ajouter_titre(carte, titre):
    titre_html = f"""
    <h3 align="center" style="
        font-size:22px;
        background-color:white;
        padding:10px;
        border:2px solid grey;">
        <b>{titre}</b>
    </h3>
    """
    carte.get_root().html.add_child(folium.Element(titre_html))

# =====================================================
# CARTE 1 : RÉPARTITION PAR TYPE D'IMPLANTATION
# =====================================================

# Création de la carte centrée sur la France
carte_implantation = folium.Map(location=[46.5, 2.5], zoom_start=6)

# Récupération des différents types d'implantation
types_implantation = df_sample["implantation_station"].unique()

# Palette de couleurs utilisée pour différencier les catégories
palette = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "lightred", "darkblue", "darkgreen",
    "cadetblue", "darkpurple", "pink",
    "lightblue", "lightgreen", "gray", "black"
]

# Attribution automatique d'une couleur à chaque type d'implantation
couleurs_implantation = {
    type_implantation: palette[i % len(palette)]
    for i, type_implantation in enumerate(types_implantation)
}

# Ajout des bornes sur la carte
for _, row in df_sample.iterrows():

    couleur = couleurs_implantation[row["implantation_station"]]

    folium.CircleMarker(
        location=[row["consolidated_latitude"],
                  row["consolidated_longitude"]],
        radius=3,
        color=couleur,
        fill=True,
        fill_color=couleur,
        fill_opacity=0.7,

        # Informations affichées lors du clic sur une borne
        popup=f"""
        <b>Type d'implantation :</b>
        {row['implantation_station']}<br>

        <b>Puissance :</b>
        {row['puissance_nominale']} kW
        """
    ).add_to(carte_implantation)

# Ajout du titre de la carte
ajouter_titre(
    carte_implantation,
    "Répartition des bornes selon le type d’implantation"
)

# Création d'une légende dynamique
# permettant d'associer chaque couleur à un type d'implantation
# ...

# Sauvegarde de la carte interactive
carte_implantation.save("carte_implantation.html")

# =====================================================
# CARTE 2 : RÉPARTITION PAR CATÉGORIE DE PUISSANCE
# =====================================================

# Création de la carte des puissances
carte_puissance = folium.Map(location=[46.5, 2.5], zoom_start=6)

# Définition des couleurs associées aux catégories de puissance
couleurs_puissance = {
    "Normale ≤ 22 kW": "blue",
    "Rapide 23-50 kW": "green",
    "Très rapide 51-150 kW": "orange",
    "Ultra rapide > 150 kW": "red"
}

# Ajout des bornes sur la carte
for _, row in df_sample.iterrows():

    couleur = couleurs_puissance[row["categorie_puissance"]]

    folium.CircleMarker(
        location=[row["consolidated_latitude"],
                  row["consolidated_longitude"]],
        radius=3,
        color=couleur,
        fill=True,
        fill_color=couleur,
        fill_opacity=0.7,

        # Informations détaillées affichées au clic
        popup=f"""
        <b>Catégorie :</b>
        {row['categorie_puissance']}<br>

        <b>Puissance :</b>
        {row['puissance_nominale']} kW
        """
    ).add_to(carte_puissance)

# Ajout du titre
ajouter_titre(
    carte_puissance,
    "Répartition des bornes selon la puissance nominale"
)

# Ajout d'une légende expliquant la signification des couleurs
# ...

# Sauvegarde de la carte
carte_puissance.save("carte_puissance.html")

# =====================================================
# CARTE 3 : HEATMAP DE DENSITÉ
# =====================================================

# Création d'une carte centrée sur la France
carte_heatmap = folium.Map(location=[46.5, 2.5], zoom_start=6)

# Extraction des coordonnées géographiques
points_heatmap = df[
    ["consolidated_latitude",
     "consolidated_longitude"]
].values.tolist()

# Génération de la carte de chaleur
# permettant de visualiser les zones de forte concentration
HeatMap(
    points_heatmap,
    radius=10,
    blur=15,
    min_opacity=0.3
).add_to(carte_heatmap)

# Ajout du titre
ajouter_titre(
    carte_heatmap,
    "Carte de chaleur de la densité des bornes de recharge"
)

# Ajout d'une légende descriptive
# ...

# Sauvegarde du résultat final
carte_heatmap.save("heatmap_irve.html")

# Message de confirmation
print("Cartes interactives générées avec succès.")
