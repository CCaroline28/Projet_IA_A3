#commande pour installer les bibliothèques : pip install pandas folium plotly scikit-learn 

import pandas as pd
import folium
from folium.plugins import HeatMap

# 1. Chargement des données
df = pd.read_csv("export_IA.csv")

# 2. Sélection des colonnes utiles
colonnes = [
    "implantation_station",
    "puissance_nominale",
    "consolidated_longitude",
    "consolidated_latitude"
]

df = df[colonnes].dropna()

# 3. Nettoyage sécurité coordonnées France
df = df[
    (df["consolidated_longitude"] >= -5.5) &
    (df["consolidated_longitude"] <= 9.5) &
    (df["consolidated_latitude"] >= 41) &
    (df["consolidated_latitude"] <= 51.5)
]

# 4. Création catégorie de puissance
def categorie_puissance(p):
    if p <= 22:
        return "Normale ≤ 22 kW"
    elif p <= 50:
        return "Rapide 23-50 kW"
    elif p <= 150:
        return "Très rapide 51-150 kW"
    else:
        return "Ultra rapide > 150 kW"

df["categorie_puissance"] = df["puissance_nominale"].apply(categorie_puissance)

# 5. Carte selon type d'implantation
carte_implantation = folium.Map(location=[46.5, 2.5], zoom_start=6)

for _, row in df.sample(min(5000, len(df)), random_state=42).iterrows():
    folium.CircleMarker(
        location=[row["consolidated_latitude"], row["consolidated_longitude"]],
        radius=3,
        popup=f"Implantation : {row['implantation_station']}",
        fill=True
    ).add_to(carte_implantation)

carte_implantation.save("carte_implantation.html")

# 6. Carte selon catégorie de puissance
# Carte selon catégorie de puissance
carte_puissance = folium.Map(location=[46.5, 2.5], zoom_start=6)

couleurs = {
    "Normale ≤ 22 kW": "blue",
    "Rapide 23-50 kW": "green",
    "Très rapide 51-150 kW": "orange",
    "Ultra rapide > 150 kW": "red"
}

for _, row in df.sample(min(5000, len(df)), random_state=42).iterrows():
    folium.CircleMarker(
        location=[row["consolidated_latitude"], row["consolidated_longitude"]],
        radius=3,
        color=couleurs[row["categorie_puissance"]],
        fill=True,
        fill_color=couleurs[row["categorie_puissance"]],
        fill_opacity=0.7,
        popup=f"Puissance : {row['puissance_nominale']} kW<br>Catégorie : {row['categorie_puissance']}"
    ).add_to(carte_puissance)

# Légende
legende_html = """
<div style="
position: fixed;
bottom: 50px;
left: 50px;
width: 230px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
font-size: 14px;
padding: 10px;
">
<b>Catégorie de puissance</b><br>
<span style="color:blue;">●</span> Normale ≤ 22 kW<br>
<span style="color:green;">●</span> Rapide 23-50 kW<br>
<span style="color:orange;">●</span> Très rapide 51-150 kW<br>
<span style="color:red;">●</span> Ultra rapide > 150 kW
</div>
"""

carte_puissance.get_root().html.add_child(folium.Element(legende_html))

carte_puissance.save("carte_puissance.html")
# 7. Heatmap densité
carte_heatmap = folium.Map(location=[46.5, 2.5], zoom_start=6)

points_heatmap = df[
    ["consolidated_latitude", "consolidated_longitude"]
].values.tolist()

HeatMap(points_heatmap, radius=10, blur=15).add_to(carte_heatmap)

carte_heatmap.save("heatmap_irve.html")

print("Cartes créées avec succès.")
