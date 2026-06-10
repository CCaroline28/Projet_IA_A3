Prédiction de la catégorie de puissance d'une borne
====================================================

Ce script prédit si une borne est Normale, Rapide, Très rapide ou Ultra rapide
à partir de ses caractéristiques (type de prise, implantation, localisation...).


Avant de commencer
------------------

Avoir export_IA.csv dans le même dossier, puis :

  pip install pandas matplotlib scikit-learn joblib


Comment lancer
--------------

  python modele_puissance.py


Ce que ça produit
-----------------

  - modele_puissance.pkl              : modèle Random Forest entraîné
  - label_encoder_puissance.pkl       : encodeur des catégories
  - confusion_matrix_puissance.png    : matrice de confusion
  - feature_importance_puissance.png  : importance des variables


Script d'exemple
----------------

import joblib
import pandas as pd

model = joblib.load("modele_puissance.pkl")
le    = joblib.load("label_encoder_puissance.pkl")

borne = pd.DataFrame([{
    "nbre_pdc"             : 2,
    "prise_type_ef"        : 0,
    "prise_type_2"         : 1,
    "prise_type_combo_ccs" : 0,
    "prise_type_chademo"   : 0,
    "prise_type_autre"     : 0,
    "gratuit"              : 1,
    "consolidated_latitude": 48.85,
    "consolidated_longitude": 2.35,
    "implantation_station_Voirie": 1,          # one-hot
    "implantation_station_Parking public": 0,
    "implantation_station_Parking prive reserve a la clientele": 0,
}])

pred = model.predict(borne)
print("Catégorie prédite :", le.inverse_transform(pred)[0])
# → ex : Normale
