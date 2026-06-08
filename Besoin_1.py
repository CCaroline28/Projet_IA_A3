#commande pour installer les bibliothèques : pip install pandas folium plotly scikit-learn 
import pandas as pd
import folium
from folium.plugins import HeatMap

# =====================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# =====================================================

# Chargement du fichier CSV nettoyé issu de la partie Big Data
df = pd.read_csv("export_IA.csv")

# Sélection des colonnes utiles pour les cartes
colonnes = [
    "implantation_station",
    "puissance_nominale",
    "consolidated_longitude",
    "consolidated_latitude"
]

# Suppression des lignes incomplètes
df = df[colonnes].dropna()

# Filtrage géographique pour conserver la France métropolitaine
df = df[
    (df["consolidated_longitude"] >= -5.5) &
    (df["consolidated_longitude"] <= 9.5) &
    (df["consolidated_latitude"] >= 41) &
    (df["consolidated_latitude"] <= 51.5)
]

# =====================================================
# CRÉATION DES CATÉGORIES DE PUISSANCE
# =====================================================

# Fonction qui transforme une puissance numérique en catégorie lisible
def categorie_puissance(p):
    if p <= 22:
        return "Normale ≤ 22 kW"
    elif p <= 50:
        return "Rapide 23-50 kW"
    elif p <= 150:
        return "Très rapide 51-150 kW"
    else:
        return "Ultra rapide > 150 kW"

# Ajout d'une colonne de catégorie de puissance
df["categorie_puissance"] = df["puissance_nominale"].apply(categorie_puissance)

# Échantillon utilisé pour alléger les cartes interactives
df_sample = df.sample(min(5000, len(df)), random_state=42)

# =====================================================
# FONCTION POUR AJOUTER UN TITRE À UNE CARTE
# =====================================================

def ajouter_titre(carte, titre):
    titre_html = f"""
    <h3 align="center" style="
        font-size:22px;
        background-color:white;
        padding:10px;
        border:2px solid grey;
        z-index:9999;">
        <b>{titre}</b>
    </h3>
    """
    carte.get_root().html.add_child(folium.Element(titre_html))


# =====================================================
# CARTE 1 : TYPE D'IMPLANTATION
# =====================================================

# Création d'une carte centrée sur la France
carte_implantation = folium.Map(location=[46.5, 2.5], zoom_start=6)

# Récupération des différents types d'implantation
types_implantation = df_sample["implantation_station"].unique()

# Palette de couleurs pour différencier les types d'implantation
palette = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "darkblue", "darkgreen", "cadetblue",
    "darkpurple", "pink", "lightblue", "lightgreen",
    "gray", "black"
]

# Attribution automatique d'une couleur différente à chaque type
couleurs_implantation = {
    type_implantation: palette[i % len(palette)]
    for i, type_implantation in enumerate(types_implantation)
}

# Ajout des points sur la carte
for _, row in df_sample.iterrows():
    couleur = couleurs_implantation[row["implantation_station"]]

    folium.CircleMarker(
        location=[
            row["consolidated_latitude"],
            row["consolidated_longitude"]
        ],
        radius=3,
        color=couleur,
        fill=True,
        fill_color=couleur,
        fill_opacity=0.7,
        popup=f"""
        <b>Type d'implantation :</b> {row['implantation_station']}<br>
        <b>Puissance nominale :</b> {row['puissance_nominale']} kW
        """
    ).add_to(carte_implantation)

# Ajout du titre
ajouter_titre(
    carte_implantation,
    "Carte des bornes selon le type d’implantation"
)

# Légende spécifique à la carte d'implantation
legende_implantation = """
<div style="
position: fixed;
bottom: 50px;
left: 50px;
width: 300px;
max-height: 320px;
overflow-y: auto;
background-color: white;
border: 2px solid grey;
z-index: 9999;
font-size: 13px;
padding: 10px;">
<b>Légende - Type d’implantation</b><br>
Chaque couleur représente un type d’implantation de station.<br><br>
"""

for type_implantation, couleur in couleurs_implantation.items():
    legende_implantation += f"""
    <span style="color:{couleur};">●</span> {type_implantation}<br>
    """

legende_implantation += """
<br>
<i>Objectif : identifier la répartition spatiale des bornes selon leur implantation.</i>
</div>
"""

carte_implantation.get_root().html.add_child(folium.Element(legende_implantation))

# Sauvegarde de la carte
carte_implantation.save("carte_implantation.html")


# =====================================================
# CARTE 2 : CATÉGORIE DE PUISSANCE
# =====================================================

# Création d'une carte centrée sur la France
carte_puissance = folium.Map(location=[46.5, 2.5], zoom_start=6)

# Couleurs fixes pour les catégories de puissance
couleurs_puissance = {
    "Normale ≤ 22 kW": "blue",
    "Rapide 23-50 kW": "green",
    "Très rapide 51-150 kW": "orange",
    "Ultra rapide > 150 kW": "red"
}

# Ajout des bornes sur la carte selon leur catégorie de puissance
for _, row in df_sample.iterrows():
    couleur = couleurs_puissance[row["categorie_puissance"]]

    folium.CircleMarker(
        location=[
            row["consolidated_latitude"],
            row["consolidated_longitude"]
        ],
        radius=3,
        color=couleur,
        fill=True,
        fill_color=couleur,
        fill_opacity=0.7,
        popup=f"""
        <b>Catégorie :</b> {row['categorie_puissance']}<br>
        <b>Puissance nominale :</b> {row['puissance_nominale']} kW
        """
    ).add_to(carte_puissance)

# Ajout du titre
ajouter_titre(
    carte_puissance,
    "Carte des bornes selon la puissance nominale"
)

# Légende spécifique à la carte de puissance
legende_puissance = """
<div style="
position: fixed;
bottom: 50px;
left: 50px;
width: 300px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
font-size: 14px;
padding: 10px;">
<b>Légende - Puissance nominale</b><br>
Les couleurs indiquent la catégorie de puissance des bornes.<br><br>
<span style="color:blue;">●</span> Normale ≤ 22 kW<br>
<span style="color:green;">●</span> Rapide 23-50 kW<br>
<span style="color:orange;">●</span> Très rapide 51-150 kW<br>
<span style="color:red;">●</span> Ultra rapide > 150 kW<br>
<br>
<i>Objectif : repérer les zones équipées en bornes rapides ou ultra-rapides.</i>
</div>
"""

carte_puissance.get_root().html.add_child(folium.Element(legende_puissance))

# Sauvegarde de la carte
carte_puissance.save("carte_puissance.html")


# =====================================================
# CARTE 3 : HEATMAP DE DENSITÉ
# =====================================================

# Création d'une carte centrée sur la France
carte_heatmap = folium.Map(location=[46.5, 2.5], zoom_start=6)

# Extraction des coordonnées pour la carte de chaleur
points_heatmap = df[
    ["consolidated_latitude", "consolidated_longitude"]
].values.tolist()

# Création de la heatmap
# Carte 3 : Heatmap de densité mieux équilibrée
carte_heatmap = folium.Map(location=[46.5, 2.5], zoom_start=6)

points_heatmap = df[
    ["consolidated_latitude", "consolidated_longitude"]
].values.tolist()

HeatMap(
    points_heatmap,
    radius=18,
    blur=22,
    min_opacity=0.35,
    max_zoom=6,
    max_val=0.6,
    gradient={
        0.2: "blue",
        0.4: "cyan",
        0.6: "lime",
        0.8: "yellow",
        1.0: "red"
    }
).add_to(carte_heatmap)
)
# Légende spécifique à la heatmap
legende_heatmap = """
<div style="
position: fixed;
bottom: 50px;
left: 50px;
width: 320px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
font-size: 14px;
padding: 10px;">
<b>Légende - Densité spatiale</b><br>
La couleur indique la concentration des bornes dans une zone.<br><br>
<span style="color:blue;">●</span> Bleu : faible densité<br>
<span style="color:limegreen;">●</span> Vert : densité moyenne<br>
<span style="color:yellow;">●</span> Jaune : densité élevée<br>
<span style="color:red;">●</span> Rouge : très forte densité<br>
<br>
<i>Objectif : identifier les zones où les bornes sont les plus concentrées.</i>
</div>
"""

carte_heatmap.get_root().html.add_child(folium.Element(legende_heatmap))

# Sauvegarde de la heatmap
carte_heatmap.save("heatmap_irve.html")

print("Cartes avec titres et légendes adaptées créées avec succès.")
