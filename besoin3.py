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
df = df.sample(n=10_000, random_state=42)

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
bool_cols = ["puissance_rapide", "prise_type_combo_ccs", "prise_type_chademo", "est_payant"]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

df = df.dropna(subset=features)
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

# Sauvegarde obligatoire pour le script final et la partie Web car le script ne doit pas relancer un apprentissage à chaque usage, il doit impérativement charger les modèles préalablement enregistrés
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
    "n_estimators": [100],
    "max_depth": [None, 10],
    "min_samples_split": [2],
}
# n_jobs=-1 pour utiliser tous les cœurs du processeur et accélérer la recherche
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
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
# Sauvegarde obligatoire pour le script final et la partie Web car le script ne doit pas relancer un apprentissage à chaque usage, il doit impérativement charger les modèles préalablement enregistrés
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
# --- Graphe 1 : distribution de la cible ---
df[cible].value_counts().plot(kind="bar", color="steelblue", figsize=(10, 5))
plt.title("Distribution des types d'implantation")
plt.xlabel("Type d'implantation")
plt.ylabel("Nombre de bornes")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("distribution_cible.png", dpi=150)
plt.close()

# =====================================================
# 8. JUSTIFICATION DES VARIABLES (tableau + graphiques)
# =====================================================

 
df_clean = df.dropna(subset=features + [cible])
 
# -------------------------------------------------------
# TABLEAU : justification du choix des variables
# -------------------------------------------------------
fig_table, ax_table = plt.subplots(figsize=(16, 5))
ax_table.axis("off")
 
colonnes = ["Variable", "Type", "Rôle attendu", "Lien avec la cible"]
lignes = [
    ["puissance_nominale",      "Numérique",  "Puissance de charge (kW)",               "Stations autoroute = puissance élevée"],
    ["puissance_rapide",        "Booléen",    "Borne rapide (oui/non)",                  "Bornes rapides surtout en voirie/autoroute"],
    ["nbre_pdc",                "Numérique",  "Nombre de points de charge",              "Parkings privés ont souvent plus de PDC"],
    ["prise_type_combo_ccs",    "Booléen",    "Prise CCS (charge rapide DC)",            "Indique une borne rapide = voirie/autoroute"],
    ["prise_type_chademo",      "Booléen",    "Prise CHAdeMO (charge rapide DC)",        "Idem, corrélé aux bornes rapides"],
    ["est_payant",              "Booléen",    "Borne payante (oui/non)",                 "Bornes de voirie souvent payantes"],
    ["consolidated_latitude",   "Numérique",  "Latitude géographique",                   "Centre-ville → voirie / périphérie → parking"],
    ["consolidated_longitude",  "Numérique",  "Longitude géographique",                  "Complément spatial de la latitude"],
]
 
table = ax_table.table(
    cellText=lignes,
    colLabels=colonnes,
    cellLoc="left",
    loc="center",
    colWidths=[0.22, 0.10, 0.30, 0.38],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)
 
# Style entêtes
for j in range(len(colonnes)):
    table[0, j].set_facecolor("#2c5f8a")
    table[0, j].set_text_props(color="white", fontweight="bold")
 
# Alternance de couleur sur les lignes
for i in range(1, len(lignes) + 1):
    for j in range(len(colonnes)):
        table[i, j].set_facecolor("#eaf2fb" if i % 2 == 0 else "white")
 
plt.title("Justification des variables sélectionnées", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("tableau_justification_variables.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tableau sauvegardé : tableau_justification_variables.png")
 
# -------------------------------------------------------
# GRAPHIQUE 1 : Importance des features (Random Forest)
# -------------------------------------------------------
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
colors = ["#2c5f8a" if importances[i] >= np.median(importances) else "#a8c8e8" for i in indices]
 
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(range(len(features)), importances[indices], color=colors, edgecolor="white", width=0.6)
ax.set_xticks(range(len(features)))
ax.set_xticklabels([features[i] for i in indices], rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Importance (Gini)", fontsize=11)
ax.set_title("Importance des variables — Random Forest", fontsize=13, fontweight="bold")
ax.axhline(np.median(importances), color="tomato", linestyle="--", linewidth=1.2, label="Médiane")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
 
# Valeurs sur les barres
for bar, idx in zip(bars, indices):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{importances[idx]:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
 
plt.tight_layout()
plt.savefig("feature_importance_implantation.png", dpi=150)
plt.close()
print("Graphique sauvegardé : feature_importance_implantation.png")
 
# -------------------------------------------------------
# GRAPHIQUE 2 : Distribution de la cible
# -------------------------------------------------------
counts = df_clean[cible].value_counts()
colors_bar = ["#2c5f8a", "#5b9ec9", "#a8c8e8"]
 
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(counts.index, counts.values, color=colors_bar, edgecolor="white", width=0.5)
ax.set_ylabel("Nombre de bornes", fontsize=11)
ax.set_title("Distribution des types d'implantation", fontsize=13, fontweight="bold")
ax.set_xticklabels(counts.index, rotation=20, ha="right", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
 
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)
 
plt.tight_layout()
plt.savefig("distribution_cible.png", dpi=150)
plt.close()
print("Graphique sauvegardé : distribution_cible.png")
 
# -------------------------------------------------------
# GRAPHIQUE 3 : Taux de bornes rapides par implantation
# -------------------------------------------------------
taux = df_clean.groupby(cible)["puissance_rapide"].mean() * 100
 
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(taux.index, taux.values, color=colors_bar, edgecolor="white", width=0.5)
ax.set_ylabel("% de bornes rapides", fontsize=11)
ax.set_title("Taux de bornes rapides par type d'implantation", fontsize=13, fontweight="bold")
ax.set_xticklabels(taux.index, rotation=20, ha="right", fontsize=10)
ax.set_ylim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
 
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
 
plt.tight_layout()
plt.savefig("taux_rapide_par_implantation.png", dpi=150)
plt.close()
print("Graphique sauvegardé : taux_rapide_par_implantation.png")
 
# -------------------------------------------------------
# GRAPHIQUE 4 : Puissance moyenne par implantation
# -------------------------------------------------------
puissance_moy = df_clean.groupby(cible)["puissance_nominale"].mean()
 
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(puissance_moy.index, puissance_moy.values, color=colors_bar, edgecolor="white", width=0.5)
ax.set_ylabel("Puissance moyenne (kW)", fontsize=11)
ax.set_title("Puissance nominale moyenne par type d'implantation", fontsize=13, fontweight="bold")
ax.set_xticklabels(puissance_moy.index, rotation=20, ha="right", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
 
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f} kW", ha="center", va="bottom", fontsize=9)
 
plt.tight_layout()
plt.savefig("puissance_par_implantation.png", dpi=150)
plt.close()
print("Graphique sauvegardé : puissance_par_implantation.png")
 
print("\nTous les graphiques de justification sont sauvegardés.")
