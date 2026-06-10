import joblib
import pandas as pd

# =====================================================
# SCRIPT DE PRÉDICTION DU CLUSTER D'UNE BORNE
# =====================================================

# Chargement du modèle K-Means déjà entraîné
# Important : on ne relance pas l'entraînement ici
modele = joblib.load("modele_kmeans.pkl")

# =====================================================
# ENTRÉE UTILISATEUR
# =====================================================

latitude = float(input("Entrez la latitude de la borne : "))
longitude = float(input("Entrez la longitude de la borne : "))

# Création d'un tableau avec les mêmes colonnes que pendant l'entraînement
nouvelle_borne = pd.DataFrame(
    [[latitude, longitude]],
    columns=[
        "consolidated_latitude",
        "consolidated_longitude"
    ]
)

# =====================================================
# PRÉDICTION
# =====================================================

cluster = modele.predict(nouvelle_borne)[0]

print("\nRésultat :")
print(f"La borne appartient au cluster {cluster}.")
