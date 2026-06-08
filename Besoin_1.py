import pandas as pd
import folium
from folium.plugins import HeatMap

# =====================================================
# 1. CHARGEMENT DES DONNÉES
# =====================================================

df = pd.read_csv("export_IA.csv")

# Conversion des coordonnées en numérique
df["consolidated_longitude"] = pd.to_numeric(
    df["consolidated_longitude"],
    errors="coerce"
)

df["consolidated_latitude"] = pd.to_numeric(
    df["consolidated_latitude"],
    errors="coerce"
)

# Suppression des coordonnées manquantes
df = df.dropna(
    subset=["consolidated_longitude", "consolidated_latitude"]
)

# Filtrage France métropolitaine
df = df[
    (df["consolidated_longitude"] >= -5.5) &
    (df["consolidated_longitude"] <= 9.5) &
    (df["consolidated_latitude"] >= 41) &
    (df["consolidated_latitude"] <= 51.5)
]

print("Nombre de points utilisés :", len(df))

# =====================================================
# 2. AGRÉGATION DES POINTS PROCHES
# =====================================================

# On regroupe les bornes proches pour obtenir une densité plus lisible
df["lat_zone"] = df["consolidated_latitude"].round(2)
df["lon_zone"] = df["consolidated_longitude"].round(2)

densite = (
    df.groupby(["lat_zone", "lon_zone"])
    .size()
    .reset_index(name="nombre_bornes")
)

# Limitation du poids des zones très denses
# Cela évite que Paris écrase visuellement les autres métropoles
seuil_max = densite["nombre_bornes"].quantile(0.85)
densite["poids"] = densite["nombre_bornes"].clip(upper=seuil_max)

print("Nombre de zones affichées :", len(densite))
print("Seuil maximum appliqué :", seuil_max)

# Format attendu par Folium HeatMap :
# [latitude, longitude, poids]
points_heatmap = densite[
    ["lat_zone", "lon_zone", "poids"]
].values.tolist()

# =====================================================
# 3. CRÉATION DE LA CARTE
# =====================================================

carte_heatmap = folium.Map(
    location=[46.5, 2.5],
    zoom_start=6,
    tiles="OpenStreetMap"
)

# =====================================================
# 4. CRÉATION DE LA HEATMAP
# =====================================================

HeatMap(
    points_heatmap,
    radius=9,
    blur=7,
    min_opacity=0.08,
    gradient={
        0.20: "blue",
        0.40: "cyan",
        0.60: "green",
        0.78: "yellow",
        0.90: "orange",
        1.00: "red"
    }
).add_to(carte_heatmap)

# =====================================================
# 5. TITRE
# =====================================================

titre_html = """
<div style="
position: fixed;
top: 10px;
left: 25%;
width: 50%;
background-color: white;
border: 2px solid grey;
z-index: 9999;
text-align: center;
font-size: 22px;
font-weight: bold;
padding: 8px;">
Carte de chaleur de la densité des bornes IRVE
</div>
"""

carte_heatmap.get_root().html.add_child(
    folium.Element(titre_html)
)

# =====================================================
# 6. LÉGENDE
# =====================================================

legende_heatmap = """
<div style="
position: fixed;
bottom: 40px;
left: 40px;
width: 350px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
font-size: 14px;
padding: 10px;">
<b>Légende - Densité spatiale</b><br><br>

La couleur représente la concentration des bornes
dans une zone géographique.<br><br>

<span style="color:blue;">●</span> Faible concentration<br>
<span style="color:cyan;">●</span> Concentration modérée<br>
<span style="color:green;">●</span> Concentration moyenne<br>
<span style="color:yellow;">●</span> Forte concentration<br>
<span style="color:orange;">●</span> Très forte concentration<br>
<span style="color:red;">●</span> Concentration maximale<br><br>

<i>
Le poids des zones très denses est limité afin de mieux faire ressortir
les autres métropoles françaises.
</i>
</div>
"""

carte_heatmap.get_root().html.add_child(
    folium.Element(legende_heatmap)
)

# =====================================================
# 7. SAUVEGARDE
# =====================================================

carte_heatmap.save("heatmap_irve.html")

print("Heatmap améliorée créée : heatmap_irve.html")
