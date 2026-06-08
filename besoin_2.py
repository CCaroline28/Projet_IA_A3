import pandas as pd
import folium
import joblib
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# =====================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =====================================================

df = pd.read_csv("export_IA.csv")

colonnes = [
    "consolidated_latitude",
    "consolidated_longitude"
]

df = df[colonnes].dropna()

df["consolidated_latitude"] = pd.to_numeric(
    df["consolidated_latitude"],
    errors="coerce"
)

df["consolidated_longitude"] = pd.to_numeric(
    df["consolidated_longitude"],
    errors="coerce"
)

df = df.dropna()

# Filtrage France métropolitaine
df = df[
    (df["consolidated_longitude"] >= -5.5) &
    (df["consolidated_longitude"] <= 9.5) &
    (df["consolidated_latitude"] >= 41) &
    (df["consolidated_latitude"] <= 51.5)
]

print("Nombre de bornes utilisées :", len(df))

X = df[
    ["consolidated_latitude", "consolidated_longitude"]
].values

# =====================================================
# 2. TEST DE PLUSIEURS PARAMÈTRES DBSCAN
# =====================================================

resultats = []

eps_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
min_samples_values = [10, 20, 30, 40]

for eps in eps_values:
    for min_samples in min_samples_values:

        modele = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = modele.fit_predict(X)

        # Nombre de clusters, en excluant le bruit (-1)
        clusters = set(labels)
        if -1 in clusters:
            clusters.remove(-1)

        nb_clusters = len(clusters)
        nb_bruit = list(labels).count(-1)

        # Les métriques ne sont calculées que si au moins 2 clusters existent
        mask = labels != -1

        if nb_clusters >= 2 and sum(mask) > 0:
            X_valid = X[mask]
            labels_valid = labels[mask]

            silhouette = silhouette_score(X_valid, labels_valid)
            calinski = calinski_harabasz_score(X_valid, labels_valid)
            davies = davies_bouldin_score(X_valid, labels_valid)
        else:
            silhouette = np.nan
            calinski = np.nan
            davies = np.nan

        resultats.append({
            "eps": eps,
            "min_samples": min_samples,
            "nb_clusters": nb_clusters,
            "nb_bruit": nb_bruit,
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies
        })

# =====================================================
# 3. TABLEAU DES RÉSULTATS
# =====================================================

resultats_df = pd.DataFrame(resultats)

print("\nRésultats des tests DBSCAN :")
print(resultats_df)

resultats_df.to_csv(
    "resultats_dbscan.csv",
    index=False
)

# Sélection du meilleur modèle :
# silhouette élevée + Davies-Bouldin faible
resultats_valides = resultats_df.dropna()

meilleur = resultats_valides.sort_values(
    by=["silhouette", "davies_bouldin"],
    ascending=[False, True]
).iloc[0]

best_eps = meilleur["eps"]
best_min_samples = int(meilleur["min_samples"])

print("\nMeilleurs paramètres trouvés :")
print("eps =", best_eps)
print("min_samples =", best_min_samples)

# =====================================================
# 4. ENTRAÎNEMENT FINAL DU MODÈLE DBSCAN
# =====================================================

dbscan_final = DBSCAN(
    eps=best_eps,
    min_samples=best_min_samples
)

df["cluster"] = dbscan_final.fit_predict(X)

# Sauvegarde du modèle
joblib.dump(dbscan_final, "modele_dbscan.pkl")

# Sauvegarde des données avec clusters
df.to_csv(
    "bornes_clusters_dbscan.csv",
    index=False
)

print("\nModèle DBSCAN sauvegardé : modele_dbscan.pkl")

# =====================================================
# 5. CRÉATION DE LA CARTE DES CLUSTERS
# =====================================================

carte_clusters = folium.Map(
    location=[46.5, 2.5],
    zoom_start=6,
    tiles="OpenStreetMap"
)

palette = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "darkblue", "darkgreen", "cadetblue",
    "darkpurple", "pink", "lightblue", "lightgreen",
    "gray", "black"
]

# Échantillon pour éviter une carte trop lourde
df_sample = df.sample(min(7000, len(df)), random_state=42)

for _, row in df_sample.iterrows():

    cluster = int(row["cluster"])

    if cluster == -1:
        couleur = "gray"
        nom_cluster = "Bruit / point isolé"
    else:
        couleur = palette[cluster % len(palette)]
        nom_cluster = f"Cluster {cluster}"

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
        <b>{nom_cluster}</b><br>
        Latitude : {row['consolidated_latitude']}<br>
        Longitude : {row['consolidated_longitude']}
        """
    ).add_to(carte_clusters)

# Titre
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
Clustering DBSCAN des bornes IRVE
</div>
"""

carte_clusters.get_root().html.add_child(
    folium.Element(titre_html)
)

# Légende
legende_html = """
<div style="
position: fixed;
bottom: 40px;
left: 40px;
width: 320px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
font-size: 14px;
padding: 10px;">
<b>Légende - Clusters DBSCAN</b><br><br>
Chaque couleur représente un cluster géographique.<br>
Les points gris correspondent aux bornes isolées.<br><br>
<i>DBSCAN regroupe les bornes selon leur densité spatiale.</i>
</div>
"""

carte_clusters.get_root().html.add_child(
    folium.Element(legende_html)
)

carte_clusters.save("carte_clusters_dbscan.html")

print("Carte créée : carte_clusters_dbscan.html")
print("Résultats sauvegardés : resultats_dbscan.csv")
print("Données clusterisées : bornes_clusters_dbscan.csv")
