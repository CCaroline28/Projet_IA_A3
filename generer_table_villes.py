import pandas as pd

# Chargement du dataset
df = pd.read_csv("export_IA.csv")

# Sélection des colonnes utiles
villes = df[
    [
        "consolidated_commune",
        "consolidated_latitude",
        "consolidated_longitude"
    ]
].dropna()

# Renommage des colonnes
villes = villes.rename(
    columns={
        "consolidated_commune": "Ville",
        "consolidated_latitude": "Latitude",
        "consolidated_longitude": "Longitude"
    }
)

# Moyenne des coordonnées si une ville apparaît plusieurs fois
villes = (
    villes.groupby("Ville", as_index=False)
    .agg(
        Latitude=("Latitude", "mean"),
        Longitude=("Longitude", "mean")
    )
)

# Tri alphabétique
villes = villes.sort_values("Ville")

# Export Excel
villes.to_excel(
    "villes_coordonnees.xlsx",
    index=False
)

print("Fichier créé : villes_coordonnees.xlsx")