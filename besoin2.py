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

df["consolidated_latitude"] = pd.to_numeric(df["consolidated_latitude"], errors="coerce")
df["consolidated_longitude"] = pd.to_numeric(df["consolidated_longitude"], errors="coerce")
df = df.dropna()

# Filtrage France métropolitaine
df = df[
    (df["consolidated_longitude"] >= -5.5) &
    (df["consolidated_longitude"] <= 9.5) &
    (df["consolidated_latitude"] >= 41) &
    (df["consolidated_latitude"] <= 51.5)
]

print("Nombre de bornes utilisées :", len(df))

# =====================================================
# 2. NORMALISATION
# =====================================================

features = ["consolidated_latitude", "consolidated_longitude"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# 3. RECHERCHE DU K OPTIMAL (SILHOUETTE)
# =====================================================

sample = X_scaled[np.random.choice(len(X_scaled), size=2000, replace=False)]

silhouette_scores = []
k_range = range(2, 11)

for k_test in k_range:
    km = KMeans(n_clusters=k_test, random_state=42, n_init=10)
    labels = km.fit_predict(sample)
    silhouette_scores.append(silhouette_score(sample, labels))

best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Meilleur k : {best_k} (score = {max(silhouette_scores):.4f})")

plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, marker='o', color='steelblue', linewidth=2)
plt.axvline(x=best_k, color='red', linestyle='--', label=f"Meilleur k = {best_k}")
plt.title("Silhouette Score selon k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.xticks(k_range)
plt.legend()
plt.savefig("silhouette_scores.png", dpi=150, bbox_inches="tight")
plt.close()

# =====================================================
# 4. CLUSTERING FINAL AVEC LE MEILLEUR K
# =====================================================

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
centers_geo = scaler.inverse_transform(kmeans.cluster_centers_)

# Graphique clustering
plt.figure(figsize=(10, 8))
for cluster_id in range(best_k):
    subset = df[df["cluster"] == cluster_id]
    plt.scatter(
        subset["consolidated_longitude"],
        subset["consolidated_latitude"],
        label=f"Cluster {cluster_id}",
        s=5,
        alpha=0.6
    )

plt.scatter(
    centers_geo[:, 1],
    centers_geo[:, 0],
    marker='X', s=200, c='red', edgecolor='black',
    zorder=5, label='Centres des clusters'
)
plt.title(f"Clustering K-Means des bornes (k={best_k})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Groupes de stations", loc="best")
plt.savefig("carte_clusteringKMeans.png", dpi=150, bbox_inches="tight")
plt.close()

# =====================================================
# 5. ANALYSE
# =====================================================

print(df["cluster"].value_counts())
print(df.groupby("cluster")[features].mean())
