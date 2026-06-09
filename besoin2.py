import pandas as pd
import folium
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
# CLUSTERING K-MEANS SELON LA POSITION 
# =====================================================

# 1. Définition des features et normalisation
features = ["consolidated_latitude", "consolidated_longitude"]
X = df[features]
# remet la latitude et longitude sur une échelle comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Entraînement du modèle
# On demande 5 groupes, il essaie 10 fois et garde le meilleur résultat
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled) # C'est cette ligne qui manquait !
# Reconvertir les centres en coordonnées géographiques réelles
centers_geo = scaler.inverse_transform(kmeans.cluster_centers_)
# centers_geo[:, 0] = latitudes, centers_geo[:, 1] = longitudes
# 3. Affichage graphique
plt.figure(figsize=(10, 8))
for cluster_id in range(k):
    subset = df[df["cluster"] == cluster_id]
    plt.scatter(
        subset["consolidated_longitude"],
        subset["consolidated_latitude"],
        label=f"Cluster {cluster_id}",
        s=5,
        alpha=0.6
    )
# Croix des centres dans le bon repère
plt.scatter(
    centers_geo[:, 1],  # longitudes
    centers_geo[:, 0],  # latitudes
    marker='X', s=200, c='red', edgecolor='black',
    zorder=5, label='Centres des clusters'
)
plt.title("Clustering K-Means des bornes selon leur position")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Groupes de stations", loc="best")
#plt.show()
plt.savefig("carte_clusteringKMeans.png", dpi=150, bbox_inches="tight")

# 4. Analyse
print(df["cluster"].value_counts())
print(df.groupby("cluster")[features].mean())