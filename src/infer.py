import os
import joblib
import pandas as pd

import sys

# Trouve la racine du projet (le dossier parent de src/)
dossier_actuel = os.path.dirname(os.path.abspath(__file__)) #__file__ contient le nom du fichier actuel (train.py), on transforme le nom du fichier en un chein complet
dossier_racine = os.path.abspath(os.path.join(dossier_actuel, ".."))#remonter jusqu'à la racine

#si le nom du dossier n'apparait pas dans le path, on l'insère au début de la liste
if dossier_racine not in sys.path:
    sys.path.insert(0, dossier_racine)

from src.data_prep import (
    Feature_engineering
)
# Configuration par défaut des chemins
MODEL_PATH = os.path.join("models", "xgb.joblib")

def run_scoring(input_csv_path, output_csv_path):
    """
    Prend un fichier CSV de clients, charge le pipeline complet,
    calcule la probabilité brute de Churn et exporte le fichier de scoring.
    """


    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Fichier d'entrée introuvable : {input_csv_path}")
        
    # Lecture des nouvelles données de production
    print(f"📖 Lecture des données depuis {input_csv_path}...")
    df_new = pd.read_csv(input_csv_path)
    
    # Chargement du pipeline complet (Pre-processing + Modèle)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Le pipeline sauvegardé est introuvable à l'emplacement : {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    #Ajout des variables Features engineering
    X, _ = Feature_engineering(df_new)
    print(f"Taille de X: {X.shape}")
    print(X.columns)
    # Calcul de la probabilité brute de Churn (classe 1)
    print("Calcul des probabilités de Churn...")
    probabilities = pipeline.predict_proba(df_new)[:, 1]
    prediction = pipeline.predict(df_new)
    # 4. Création du fichier de sortie épuré pour le Marketing
    scoring_df = pd.DataFrame({
        'customerID': df_new['customerID'],
        'proba_churn': probabilities,
        'label_pred': prediction
    })
    print(scoring_df.columns)
    # Récupération de l'identifiant s'il est présent
    #if 'customerID' in df_new.columns:
     #   scoring_df.insert(0, 'customerID', df_new['customerID'])
    
    # Exportation du fichier final
    scoring_df.to_csv(output_csv_path, index=False)
    print(f"✅ Fichier de scoring généré avec succès : {output_csv_path}")
    
    return scoring_df

def simuler_data_input():

 # Simulation de deux NOUVEAUX clients arrivant le mois prochain (SANS la colonne Churn)
 new_data = pd.DataFrame([
    {
        'customerID': '0001-FIBRE-RISK',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,                          # Client très récent
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',       # Fibre optique (Alerte prix)
        'OnlineSecurity': 'No',                # Pas de support (Alerte risque)
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',                   # Pas de support (Alerte risque)
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',          # Sans engagement (Alerte risque)
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',   # Chèque électronique (Alerte risque)
        'MonthlyCharges': 95.50,               # Panier très lourd
        'TotalCharges': 191.00
    },
    {
        'customerID': '0002-DSL-LOYAL',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 60,                         # Client historique fidélisé
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',              # DSL économique
        'OnlineSecurity': 'Yes',               # Options de sécurité actives
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',                  # Support technique actif
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Two year',                # Engagé sur 2 ans
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Credit card (automatic)', # Prélèvement automatique
        'MonthlyCharges': 45.00,
        'TotalCharges': 2700.00
    }
])

# Sauvegarde du fichier de test
 new_data.to_csv("data/raw/raw_new_customers.csv", index=False)
 print("✅ Fichier de test simulé généré dans data/raw/raw_new_customers.csv")

if __name__ == "__main__":
    # Chemins de simulation pour tester le script indépendamment
    INPUT_SIMULATION = os.path.join("data/raw", "raw_new_customers.csv")
    OUTPUT_SIMULATION = os.path.join("data/raw", "new_data_resultats_scores.csv")
    print(f"chemin input: {INPUT_SIMULATION}")
    print(f"chemin input: {OUTPUT_SIMULATION}")
    print("Script infer.py (Scoring Pur) initialisé.")
    simuler_data_input()
    # Test et sauvegarde des résultats
    scoring_df = run_scoring(INPUT_SIMULATION, OUTPUT_SIMULATION)
