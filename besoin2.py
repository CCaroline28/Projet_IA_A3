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

df = pd.read_csv("export_IA.csv")  # Charge le fichier CSV dans un DataFrame

# On ne garde que les colonnes utiles : latitude et longitude
colonnes = [
    "consolidated_latitude",
    "consolidated_longitude"
]

df = df[colonnes].dropna()  # Supprime les lignes avec des valeurs manquantes

# Conversion explicite en nombres (au cas où les valeurs seraient du texte)
# errors="coerce" transforme les valeurs non-convertibles en NaN au lieu de planter
df["consolidated_latitude"] = pd.to_numeric(df["consolidated_latitude"], errors="coerce")
df["consolidated_longitude"] = pd.to_numeric(df["consolidated_longitude"], errors="coerce")
df = df.dropna()  # Re-supprime les NaN créés par la conversion

# Filtre géographique : on garde uniquement la France métropolitaine
# en définissant des bornes min/max pour la longitude et la latitude
df = df[
    (df["consolidated_longitude"] >= -5.5) &   # Pas trop à l'ouest (Bretagne)
    (df["consolidated_longitude"] <= 9.5) &    # Pas trop à l'est (Alsace)
    (df["consolidated_latitude"] >= 41) &      # Pas trop au sud (Corse)
    (df["consolidated_latitude"] <= 51.5)      # Pas trop au nord (Nord-Pas-de-Calais)
]

print("Nombre de bornes utilisées :", len(df))

# =====================================================
# 2. NORMALISATION
# =====================================================

features = ["consolidated_latitude", "consolidated_longitude"]
X = df[features]

# StandardScaler centre les données (moyenne = 0) et les réduit (écart-type = 1)
# Indispensable pour K-Means qui est sensible aux échelles :
# sans ça, la longitude pèserait différemment de la latitude
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit() calcule moyenne/écart-type, transform() applique

# =====================================================
# 3. RECHERCHE DU K OPTIMAL (SILHOUETTE)
# =====================================================

# Échantillon aléatoire de 2000 points pour aller plus vite
# (tester tous les k sur des milliers de bornes serait trop lent)
sample = X_scaled[np.random.choice(len(X_scaled), size=2000, replace=False)]

silhouette_scores = []
k_range = range(2, 11)  # On teste k de 2 à 10 clusters

for k_test in k_range:
    km = KMeans(n_clusters=k_test, random_state=42, n_init=10)
    # random_state=42 : résultat reproductible
    # n_init=10 : K-Means est relancé 10 fois, on garde le meilleur résultat
    labels = km.fit_predict(sample)  # Assigne chaque point à un cluster

    # Le silhouette score va de -1 à 1 :
    # proche de 1 = les clusters sont bien séparés et cohérents
    # proche de 0 = les clusters se chevauchent
    # négatif = les points sont mal classés
    silhouette_scores.append(silhouette_score(sample, labels))

# On choisit le k qui maximise le silhouette score
best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Meilleur k : {best_k} (score = {max(silhouette_scores):.4f})")

# Graphique pour visualiser l'évolution du score selon k
plt.figure(figsize=(10, 5))
plt.plot(k_range, silhouette_scores, marker='o', color='steelblue', linewidth=2)
plt.axvline(x=best_k, color='red', linestyle='--', label=f"Meilleur k = {best_k}")
plt.title("Silhouette Score selon k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.xticks(k_range)
plt.legend()
plt.savefig("silhouette_scores.png", dpi=150, bbox_inches="tight")  # Sauvegarde en image
plt.close()  # Ferme la figure pour libérer la mémoire

# =====================================================
# 4. CLUSTERING FINAL AVEC LE MEILLEUR K
# =====================================================

# On entraîne K-Means sur TOUTES les données (pas juste l'échantillon) avec le meilleur k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)  # Ajoute une colonne "cluster" au DataFrame

# inverse_transform() reconvertit les centres (normalisés) en lat/lon réelles
centers_geo = scaler.inverse_transform(kmeans.cluster_centers_)

# Graphique : chaque cluster avec une couleur différente
plt.figure(figsize=(10, 8))
for cluster_id in range(best_k):
    subset = df[df["cluster"] == cluster_id]  # Points appartenant à ce cluster
    plt.scatter(
        subset["consolidated_longitude"],
        subset["consolidated_latitude"],
        label=f"Cluster {cluster_id}",
        s=5,       # Petits points (beaucoup de bornes)
        alpha=0.6  # Semi-transparent pour mieux voir la densité
    )

# Affiche les centres des clusters en rouge avec un marqueur X
plt.scatter(
    centers_geo[:, 1],  # colonne 1 = longitude
    centers_geo[:, 0],  # colonne 0 = latitude
    marker='X', s=200, c='red', edgecolor='black',
    zorder=5,           # Affiché au-dessus des autres points
    label='Centres des clusters'
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

carte.get_root().html.add_child(folium.Element(legende_html))

carte.save("carte_clusters_kmeans.html")

print("\nCarte créée : carte_clusters_kmeans.html")
print("CSV créé : bornes_clusters_kmeans.csv")
