# Clustering géographique des bornes de recharge

Ce script regroupe les bornes de recharge en clusters géographiques avec K-Means,
génère une carte interactive et permet de prédire à quel groupe appartient une nouvelle borne.

## Pour lancer

Avoir `export_IA.csv` dans le même dossier, puis :

```bash
pip install pandas folium scikit-learn matplotlib joblib openpyxl
python clustering_kmeans.py
```

## Ce que ça produit

| Fichier | Description |
|---|---|
| `modele_kmeans.pkl` | Modèle K-Means entraîné |
| `carte_clusteringKMeans.png` | Carte statique des clusters |
| `carte_clusters_kmeans.html` | Carte interactive (à ouvrir dans un navigateur) |
| `bornes_clusters_kmeans.csv` | Dataset avec le cluster de chaque borne |
| `villes_coordonnees.xlsx` | Coordonnées moyennes par ville |

## Prédire le cluster d'une nouvelle borne

```python
import joblib
import pandas as pd

modele = joblib.load("modele_kmeans.pkl")

nouvelle_borne = pd.DataFrame(
    [[48.85, 2.35]],
    columns=["consolidated_latitude", "consolidated_longitude"]
)

cluster = modele.predict(nouvelle_borne)[0]
print(f"La borne appartient au cluster {cluster}.")
```

> Le modèle utilise 5 clusters déterminés par la méthode du coude
> et validés avec les scores Silhouette, Calinski-Harabasz et Davies-Bouldin.
