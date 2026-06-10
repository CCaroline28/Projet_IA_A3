Prédiction d'implantation de bornes de recharge
================================================

Ce script prédit si une borne se trouve en voirie, parking public
ou parking privé à partir de ses caractéristiques techniques.

Avant de commencer, assure-toi d'avoir les trois fichiers .pkl générés
par l'entraînement dans le même dossier :
  - modele_implantation.pkl
  - scaler_implantation.pkl
  - label_encoder_implantation.pkl


Comment utiliser le script final
---------------------------------

1. Installe les dépendances si ce n'est pas déjà fait :
   pip install pandas scikit-learn joblib

2. Lance le script :
   python predict_implantation.py


Script d'exemple (predict_implantation.py)
------------------------------------------

import joblib
import pandas as pd

# Chargement des fichiers entraînés
model  = joblib.load("modele_implantation.pkl")
scaler = joblib.load("scaler_implantation.pkl")
le     = joblib.load("label_encoder_implantation.pkl")

# Renseigne ici les caractéristiques de la borne à prédire
borne = pd.DataFrame([{
    "puissance_nominale"   : 22,    # puissance en kW
    "puissance_rapide"     : 0,     # 1 si borne rapide, 0 sinon
    "nbre_pdc"             : 4,     # nombre de points de charge
    "prise_type_combo_ccs" : 0,     # 1 si prise CCS, 0 sinon
    "prise_type_chademo"   : 0,     # 1 si prise CHAdeMO, 0 sinon
    "est_payant"           : 1,     # 1 si payant, 0 sinon
    "consolidated_latitude": 48.85,
    "consolidated_longitude": 2.35,
}])

# Prédiction
prediction = model.predict(scaler.transform(borne))
print("Type d'implantation prédit :", le.inverse_transform(prediction)[0])
# → ex : Parking public
