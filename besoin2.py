################## K - means ( Manon )
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














######### K - means ( Aya ) ##########
import pandas as pd

# =====================================================
# 1. CHARGEMENT DES DONNÉES
# =====================================================

# Chargement du fichier CSV nettoyé
df = pd.read_csv("export_IA.csv")

print("Nombre de lignes initial :", len(df))

# =====================================================
# 2. SÉLECTION DES VARIABLES
# =====================================================

# Pour le clustering géographique,
# seules les coordonnées sont nécessaires

df = df[
    [
        "consolidated_latitude",
        "consolidated_longitude"
    ]
]

# =====================================================
# 3. NETTOYAGE
# =====================================================

# Suppression des lignes avec coordonnées manquantes

df = df.dropna()

# Conversion en format numérique

df["consolidated_latitude"] = pd.to_numeric(
    df["consolidated_latitude"],
    errors="coerce"
)

df["consolidated_longitude"] = pd.to_numeric(
    df["consolidated_longitude"],
    errors="coerce"
)

df = df.dropna()

# =====================================================
# 4. FILTRAGE FRANCE MÉTROPOLITAINE
# =====================================================

df = df[
    (df["consolidated_longitude"] >= -5.5) &
    (df["consolidated_longitude"] <= 9.5) &
    (df["consolidated_latitude"] >= 41) &
    (df["consolidated_latitude"] <= 51.5)
]

print("Nombre de lignes après nettoyage :", len(df))

# =====================================================
# 5. MATRICE POUR K-MEANS
# =====================================================

X = df[
    [
        "consolidated_latitude",
        "consolidated_longitude"
    ]
]

print("\nAperçu des données utilisées :")
print(X.head())

print("\nDimensions de la matrice :")
print(X.shape)

# =====================================================
# 6. MÉTHODE DU COUDE
# =====================================================

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inerties = []

K_range = range(2, 16)

for k in K_range:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    inerties.append(kmeans.inertia_)

# =====================================================
# 7. GRAPHIQUE DU COUDE
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    K_range,
    inerties,
    marker="o"
)

plt.title("Méthode du coude")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertie")

plt.grid(True)

plt.show()
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

print("\nÉvaluation des différents K :\n")

for k in range(2, 16):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    silhouette = silhouette_score(X, labels)

    calinski = calinski_harabasz_score(
        X,
        labels
    )

    davies = davies_bouldin_score(
        X,
        labels
    )

    print(
        f"K={k} | "
        f"Silhouette={silhouette:.4f} | "
        f"Calinski={calinski:.2f} | "
        f"Davies={davies:.4f}"
    )
    import folium

# =====================================================
# 8. K-MEANS FINAL AVEC K OPTIMAL
# =====================================================

kmeans_final = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans_final.fit_predict(X)
import joblib

joblib.dump(kmeans_final, "modele_kmeans.pkl")

print("Modèle sauvegardé : modele_kmeans.pkl")

print("\nRépartition des bornes par cluster :")
print(df["cluster"].value_counts().sort_index())


print("\nRépartition des bornes par cluster :")
print(df["cluster"].value_counts().sort_index())

df.to_csv("bornes_clusters_kmeans.csv", index=False)

# =====================================================
# 9. CARTE DES CLUSTERS
# =====================================================

carte = folium.Map(
    location=[46.5, 2.5],
    zoom_start=6,
    tiles="OpenStreetMap"
)

couleurs_clusters = {
    0: "red",
    1: "blue",
    2: "green",
    3: "orange",
    4: "purple"
}

df_sample = df.sample(min(7000, len(df)), random_state=42)

for _, row in df_sample.iterrows():
    cluster = int(row["cluster"])
    couleur = couleurs_clusters.get(cluster, "gray")

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
        <b>Cluster :</b> {cluster}<br>
        <b>Latitude :</b> {row['consolidated_latitude']}<br>
        <b>Longitude :</b> {row['consolidated_longitude']}
        """
    ).add_to(carte)

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
Carte des clusters K-Means des bornes IRVE
</div>
"""

carte.get_root().html.add_child(folium.Element(titre_html))

legende_html = """
<div style="
position: fixed;
bottom: 40px;
left: 40px;
width: 300px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
font-size: 14px;
padding: 10px;">
<b>Légende - Clusters K-Means</b><br><br>
<span style="color:red;">●</span> Cluster 0<br>
<span style="color:blue;">●</span> Cluster 1<br>
<span style="color:green;">●</span> Cluster 2<br>
<span style="color:orange;">●</span> Cluster 3<br>
<span style="color:purple;">●</span> Cluster 4<br><br>
<i>Chaque couleur représente un regroupement géographique de bornes.</i>
</div>
"""
import matplotlib.pyplot as plt

# Initialisation des listes
k_range = range(2, 16)
inerties = []
silhouettes = []
calinski_scores = []

# Calcul des métriques pour chaque K
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inerties.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    calinski_scores.append(calinski_harabasz_score(X_scaled, labels))

# Création de la figure avec 3 graphiques
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Méthode du Coude
axes[0].plot(k_range, inerties, marker='o', color='tab:blue')
axes[0].set_title("Méthode du Coude (Inertie)")
axes[0].set_xlabel("Nombre de clusters")
axes[0].grid(True)

# 2. Score de Silhouette
axes[1].plot(k_range, silhouettes, marker='s', color='tab:green')
axes[1].set_title("Score de Silhouette")
axes[1].set_xlabel("Nombre de clusters")
axes[1].grid(True)

# 3. Score de Calinski-Harabasz
axes[2].plot(k_range, calinski_scores, marker='^', color='tab:red')
axes[2].set_title("Score de Calinski-Harabasz")
axes[2].set_xlabel("Nombre de clusters")
axes[2].grid(True)

plt.tight_layout()
plt.show()
carte.get_root().html.add_child(folium.Element(legende_html))

carte.save("carte_clusters_kmeans.html")

print("\nCarte créée : carte_clusters_kmeans.html")
print("CSV créé : bornes_clusters_kmeans.csv")
