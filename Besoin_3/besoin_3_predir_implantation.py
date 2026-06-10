import joblib
import pandas as pd
import numpy as np

def predir_implantation(nouvelle_borne_data):
    """
    Prend en entrée un dictionnaire contenant les caractéristiques de la borne
    et retourne la prédiction du type d'implantation.
    """
    
    # 1. Chargement des modèles et outils sauvegardés
    try:
        le = joblib.load("label_encoder_implantation.pkl")
        scaler = joblib.load("scaler_implantation.pkl")
        model = joblib.load("modele_implantation.pkl")
    except FileNotFoundError as e:
        return f"Erreur : Fichiers de modèle non trouvés. {e}"

    # 2. Conversion des données en DataFrame
    df_input = pd.DataFrame([nouvelle_borne_data])
    
    # Ordre des colonnes tel qu'attendu par le modèle
    features_order = [
        "puissance_nominale", "puissance_rapide", "nbre_pdc", 
        "prise_type_combo_ccs", "prise_type_chademo", "est_payant", 
        "consolidated_latitude", "consolidated_longitude"
    ]
    
    # Vérification des colonnes
    df_input = df_input[features_order]

    # 3. Normalisation (on utilise le même scaler que lors de l'entraînement)
    X_scaled = scaler.transform(df_input)

    # 4. Prédiction
    prediction_index = model.predict(X_scaled)[0]
    
    # 5. Décodage du résultat
    resultat = le.inverse_transform([prediction_index])[0]
    
    return resultat

# =====================================================
# EXEMPLE D'UTILISATION
# =====================================================
if __name__ == "__main__":
    # Spécificités d'une borne exemple
    nouvelle_borne = {
        "puissance_nominale": 50.0,
        "puissance_rapide": 1,
        "nbre_pdc": 2,
        "prise_type_combo_ccs": 1,
        "prise_type_chademo": 1,
        "est_payant": 1,
        "consolidated_latitude": 48.11,
        "consolidated_longitude": -1.67
    }

    prediction = predir_implantation(nouvelle_borne)
    print(f"La borne est prédite comme étant : {prediction}")