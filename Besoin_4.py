#To do:
#Regarder toutes les colonnes disponibles
#Supprimer celles qui sont construites à partir de la puissance 
#Garder seulement les informations qu'on connaîtrait avant de connaître la puissance


import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. CHARGEMENT DES DONNÉES
# =====================================================

df = pd.read_csv("export_IA.csv")

# =====================================================
# 2. CRÉATION DE LA CIBLE
# =====================================================

def categorie_puissance(p):

    if p <= 22:
        return "Normale"

    elif p <= 50:
        return "Rapide"

    elif p <= 150:
        return "Très rapide"

    else:
        return "Ultra rapide"

df["categorie_puissance"] = (
    df["puissance_nominale"]
    .apply(categorie_puissance)
)

# =====================================================
# 3. ANALYSE DE LA RÉPARTITION
# =====================================================

print("\nRépartition des catégories :\n")

print(
    df["categorie_puissance"]
    .value_counts()
)

print("\nPourcentage :\n")

print(
    round(
        df["categorie_puissance"]
        .value_counts(normalize=True) * 100,
        2
    )
)
features = [
    "implantation_station",
    "nbre_pdc",
    "prise_type_ef",
    "prise_type_2",
    "prise_type_combo_ccs",
    "prise_type_chademo",
    "prise_type_autre",
    "gratuit",
    "consolidated_latitude",
    "consolidated_longitude"
]

colonnes_utiles = features + ["categorie_puissance"]

df = df[colonnes_utiles].dropna()

print("\nTaille du dataset :", len(df))
print("\nColonnes retenues :")
print(df.columns)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# =====================================================
# 4. ENCODAGE DES VARIABLES
# =====================================================

# Transformation de la variable texte implantation_station
# en variables numériques (One-Hot Encoding)

# =====================================================
# 4. ENCODAGE DES VARIABLES
# =====================================================
df["implantation_station"] = (
    df["implantation_station"]
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
)
X = pd.get_dummies(
    df[features],
    columns=["implantation_station"]
)

# Nettoyage des booléens écrits sous forme de texte
X = X.replace({
    "true": 1,
    "false": 0,
    "True": 1,
    "False": 0,
    True: 1,
    False: 0
})

# Conversion forcée de toutes les colonnes en numérique
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Suppression des lignes qui auraient encore des valeurs invalides
X = X.dropna()

# On aligne df avec les lignes conservées dans X
df = df.loc[X.index]

# Conversion finale
X = X.astype(float)

print("\nNombre de variables après encodage :")
print(X.shape[1])
# =====================================================
# 5. ENCODAGE DE LA CIBLE
# =====================================================

le = LabelEncoder()

y = le.fit_transform(
    df["categorie_puissance"]
)

# Sauvegarde du LabelEncoder
joblib.dump(
    le,
    "label_encoder_puissance.pkl"
)

print("\nClasses de puissance :")
print(le.classes_)

# =====================================================
# 6. TRAIN / TEST
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTaille train :", len(X_train))
print("Taille test :", len(X_test))

print("\nNombre de variables après encodage :")
print(X.shape[1])
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# =====================================================
# 7. RÉGRESSION LOGISTIQUE
# =====================================================

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n==========================")
print("RÉGRESSION LOGISTIQUE")
print("==========================")

print(
    "Accuracy :",
    accuracy_score(y_test, y_pred_lr)
)

print(
    classification_report(
        y_test,
        y_pred_lr,
        target_names=le.classes_
    )
)

# =====================================================
# 8. RANDOM FOREST
# =====================================================

rf = RandomForestClassifier(
random_state=42)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n==========================")
print("RANDOM FOREST")
print("==========================")

print(
    "Accuracy :",
    accuracy_score(y_test, y_pred_rf)
)

print(
    classification_report(
        y_test,
        y_pred_rf,
        target_names=le.classes_
    )
)
from sklearn.model_selection import GridSearchCV

# =====================================================
# 9. OPTIMISATION DU RANDOM FOREST AVEC GRIDSEARCHCV
# =====================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n==========================")
print("GRIDSEARCHCV - RANDOM FOREST")
print("==========================")

print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score CV :", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

print("\nAccuracy du meilleur modèle :")
print(accuracy_score(y_test, y_pred_best))

print(classification_report(y_test,y_pred_best,target_names=le.classes_))
import joblib

joblib.dump(
    best_model,
    "modele_puissance.pkl"
)

joblib.dump(
    le,
"label_encoder_puissance.pkl")

print("Modèle sauvegardé.")
from sklearn.metrics import (confusion_matrix,ConfusionMatrixDisplay)

cm = confusion_matrix(
    y_test,
    y_pred_best
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(ax=ax)

plt.title("Matrice de confusion - Catégorie de puissance")

plt.tight_layout()

plt.savefig("confusion_matrix_puissance.png",dpi=150)

plt.close()
import numpy as np

# =====================================================
# IMPORTANCE DES VARIABLES (VERSION AMÉLIORÉE)
# =====================================================

importances = best_model.feature_importances_

importance_df = pd.DataFrame({
    "Variable": X.columns,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(12, 8))

plt.barh(
    importance_df["Variable"],
    importance_df["Importance"]
)

plt.xlabel("Importance")

plt.title(
    "Importance des variables pour la prédiction de la catégorie de puissance"
)

plt.tight_layout()

plt.savefig(
    "feature_importance_puissance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()