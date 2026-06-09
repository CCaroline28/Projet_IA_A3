import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# =====================================================
# 1. CHARGEMENT DES DONNÉES (déjà nettoyées)
# =====================================================

df = pd.read_csv("export_IA.csv")

# Features choisies :
# - puissance_nominale : une station rapide (autoroute) a une puissance élevée
# - puissance_rapide : booléen, indique si c'est une borne rapide
# - categorie_puissance : catégorie de puissance (lente/rapide/très rapide)
# - nbre_pdc : les parkings privés ont souvent plus de bornes
# - prise_type_combo_ccs / prise_type_chademo : les bornes rapides ont ces prises
# - est_payant : les bornes de voirie sont souvent payantes
# - consolidated_latitude / consolidated_longitude : la localisation géographique
#   influence fortement le type (voirie en centre-ville, parking en périphérie)

features = [
    "puissance_nominale",
    "puissance_rapide",
    "nbre_pdc",
    "prise_type_combo_ccs",
    "prise_type_chademo",
    "est_payant",
    "consolidated_latitude",
    "consolidated_longitude",
]
cible = "implantation_station"
# On garde uniquement les colonnes utiles et on drop les lignes
colonnes_utiles = features + [cible]
# On drop les lignes avec des valeurs manquantes dans ces colonnes
df = df[colonnes_utiles].dropna()

# Conversion en numérique au cas où
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

print("Taille du dataset :", len(df))
print("Distribution de la cible :\n", df[cible].value_counts())

# =====================================================
# 2. ENCODAGE DE LA CIBLE (texte → nombre)
# =====================================================
# "Voirie" → 0, "Parking privé réservé à la clientèle" → 1, "Parking public" → 2
le = LabelEncoder()
df["cible_encodee"] = le.fit_transform(df[cible])

# Sauvegarde obligatoire pour le script final et la partie Web
joblib.dump(le, "label_encoder_implantation.pkl")
print("Classes :", le.classes_)

# =====================================================
# 3. SÉPARATION TRAIN / TEST
# =====================================================
# On sépare les features (X) de la cible (y)
X = df[features]
y = df["cible_encodee"]

# stratify=y pour garder les mêmes proportions de classes dans train et test
# important ici car "Parking privé réservé à la clientèle" est sous-représenté (424 cas)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 4. NORMALISATION
# =====================================================
# La normalisation est importante pour les modèles basés sur les distances (KNN, SVM) ou les arbres (Random Forest)
scaler = StandardScaler()
# On ajuste le scaler sur les données d'entraînement et on transforme à la fois train et test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde obligatoire pour le script final et la partie Web
joblib.dump(scaler, "scaler_implantation.pkl")

# =====================================================
# 5. COMPARAISON DE DEUX MODÈLES
# =====================================================
# On compare un modèle simple (Régression Logistique) à un modèle plus complexe (Random Forest)
# --- Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print("\n=== Random Forest ===")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# --- Régression Logistique ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("\n=== Régression Logistique ===")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

# =====================================================
# 6. OPTIMISATION AVEC GRIDSEARCHCV (sur Random Forest)
# =====================================================
# On teste différentes combinaisons d'hyperparamètres pour trouver la meilleure configuration
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}
# n_jobs=-1 pour utiliser tous les cœurs du processeur et accélérer la recherche
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
# On lance la recherche sur les données d'entraînement
grid_search.fit(X_train_scaled, y_train)

print("\nMeilleurs hyperparamètres :", grid_search.best_params_)
print("Meilleur score CV :", grid_search.best_score_)
# On évalue le meilleur modèle trouvé sur les données de test
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

print("\n=== Meilleur modèle (après GridSearch) ===")
print("Accuracy :", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best, target_names=le.classes_))

# =====================================================
# 7. SAUVEGARDE DU MODÈLE FINAL
# =====================================================
# On sauvegarde le meilleur modèle pour pouvoir l'utiliser dans le script final et la partie Web
joblib.dump(best_model, "modele_implantation.pkl")
print("Modèle sauvegardé : modele_implantation.pkl")

# =====================================================
# 8. VISUALISATIONS
# =====================================================

# --- Matrice de confusion ---
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, xticks_rotation=45)
plt.title("Matrice de confusion - Implantation station")
plt.tight_layout()
plt.savefig("confusion_matrix_implantation.png", dpi=150)
plt.close()

# --- Importance des features ---
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(features)), importances[indices], color="steelblue")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=30, ha="right")
plt.title("Importance des features - Random Forest")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_implantation.png", dpi=150)
plt.close()

print("Graphiques sauvegardés.")