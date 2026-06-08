import pandas as pd
import folium
from folium.plugins import HeatMap

# Chargement du fichier CSV nettoyé issu de la partie Big Data
df = pd.read_csv("export_IA.csv")

# Sélection des colonnes nécessaires pour le besoin 1
colonnes = [
    "implantation_station",
    "puissance_nominale",
    "consolidated_longitude",
    "consolidated_latitude"
]

# Suppression des lignes avec valeurs manquantes sur ces colonnes
df = df[colonnes].dropna()

# Filtrage des coordonnées pour garder uniquement les points situés en France métropolitaine
df = df[
    (df["consolidated_longitude"] >= -5.5) &
    (df["consolidated_longitude"] <= 9.5) &
    (df["consolidated_latitude"] >= 41) &
    (df["consolidated_latitude"] <= 51.5)
]

# Fonction permettant de transformer une puissance numérique en catégorie
def categorie_puissance(p):
    if p <= 22:
        return "Normale ≤ 22 kW"
    elif p <= 50:
        return "Rapide 23-50 kW"
    elif p <= 150:
        return "Très rapide 51-150 kW"
    else:
        return "Ultra rapide > 150 kW"

# Création d'une nouvelle colonne contenant la catégorie de puissance
df["categorie_puissance"] = df["puissance_nominale"].apply(categorie_puissance)

# Création d'une carte centrée sur la France
carte_implantation = folium.Map(location=[46.5, 2.5], zoom_start=6)

# Ajout des bornes sur la carte selon leur type d'implantation
# Un échantillon de 5000 lignes est utilisé pour éviter une carte trop lourde
for _, row in df.sample(min(5000, len(df)), random_state=42).iterrows():
    folium.CircleMarker(
        location=[row["consolidated_latitude"], row["consolidated_longitude"]],
        radius=3,
        popup=f"Implantation : {row['implantation_station']}",
        fill=True
    ).add_to(carte_implantation)

# Sauvegarde de la carte au format HTML
carte_implantation.save("carte_implantation.html")

# Création d'une deuxième carte centrée sur la France
carte_puissance = folium.Map(location=[46.5, 2.5], zoom_start=6)

# Association d'une couleur à chaque catégorie de puissance
couleurs = {
    "Normale ≤ 22 kW": "blue",
    "Rapide 23-50 kW": "green",
    "Très rapide 51-150 kW": "orange",
    "Ultra rapide > 150 kW": "red"
}

# Ajout des bornes sur la carte avec une couleur selon leur catégorie de puissance
for _, row in df.sample(min(5000, len(df)), random_state=42).iterrows():
    folium.CircleMarker(
        location=[row["consolidated_latitude"], row["consolidated_longitude"]],
        radius=3,
        color=couleurs[row["categorie_puissance"]],
        fill=True,
        fill_color=couleurs[row["categorie_puissance"]],
        popup=f"Puissance : {row['puissance_nominale']} kW<br>Catégorie : {row['categorie_puissance']}"
    ).add_to(carte_puissance)

# Sauvegarde de la carte des puissances
carte_puissance.save("carte_puissance.html")

# Création de la carte de chaleur
carte_heatmap = folium.Map(location=[46.5, 2.5], zoom_start=6)

# Récupération des coordonnées sous forme de liste
points_heatmap = df[
    ["consolidated_latitude", "consolidated_longitude"]
].values.tolist()

# Ajout de la heatmap pour visualiser les zones de forte densité
HeatMap(points_heatmap, radius=10, blur=15).add_to(carte_heatmap)

# Sauvegarde de la heatmap
carte_heatmap.save("heatmap_irve.html")

# Message de confirmation
print("Cartes créées avec succès.")