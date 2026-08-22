# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, recall_score, precision_score
from typing import Any, Tuple
import numpy as np

#pour le preprosessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import sys
import os
# Trouve la racine du projet (le dossier parent de src/)
dossier_actuel = os.path.dirname(os.path.abspath(__file__)) #__file__ contient le nom du fichier actuel (train.py), on transforme le nom du fichier en un chein complet
dossier_racine = os.path.abspath(os.path.join(dossier_actuel, ".."))#remonter jusqu'à la racine

#si le nom du dossier n'apparait pas dans le path, on l'insère au début de la liste
if dossier_racine not in sys.path:
    sys.path.insert(0, dossier_racine)

# Importer les fonctions à partir de src.data_prep.py
#from src.data_prep import obtenir_preprocess
from src.data_prep import load_data
from src.data_prep import get_column_types
from src.data_prep import features_target_separation
from sklearn.model_selection import RandomizedSearchCV


def data_split_test_train(X:pd.DataFrame, y:pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    '''
    Sépare les données en ensembles d'entraînement et de test.
    variables explicatives (X) et variable cible (y).
    Parameters  
    ----------
    X : pd.DataFrame
        Variables explicatives.
    y : pd.Series
        Variable cible.
    Returns
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test, id_test
    -------
    '''

    #3. Split des données en train/test
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #4. Extraction sécurisée des IDs pour le fichier de scoring
    id_test = X_test_full['customerID']

    #5. Suppression de la colonne qui ne sera pas inclue dans le calcul ML
    X_train = X_train_full.drop(columns=['customerID'])
    X_test = X_test_full.drop(columns=['customerID'])

    return X_train, X_test, y_train, y_test, id_test

# 6. Preporcessing

#preprocesseur = obtenir_preprocess(X_train)



# 5. (Ensuite on peut définir les fonctions pour entraîner les modèles...)
def entrainer_les_modeles(preprocesseur: Any, estimator: Any, X:pd.DataFrame, y:pd.Series, **kwargs):
    """
    Crée le pipeline final avec le préprocesseur et l'estimateur fournis,
    puis entraîne le modèle sur les données X et y.

    Parameters
    ----------
    preprocessur : Pipeline ou ColumnTransformer (dans notre cas)
        Modèle entraîné (pipeline scikit-learn).
    X_train : pd.DataFrame
        Variables explicatives du jeu d'entraînement.
    y_train : pd.Series
        Variable cible du jeu d'entraînement.

    **kwargs : dict
        Arguments supplémentaires passés à la méthode fit.
    Returns
    -------
    BaseEstimator : retounre le modèle
    """
    # 1. Construction de la pipeline finale
    model = Pipeline(
        steps=[
            ("preprocessor", preprocesseur),
            ("estimator", estimator)
        ]
    )
    # 2. Entraînement et retour du modèle entraîné
    return model.fit(X, y, **kwargs)

def assemblage_pipeline_global(preprocesseur: Any, set_seed):
 # Assemblage du Pipeline global
 # Le pipeline applique les transformations, filtre via RFECV, puis entraîne le modèle
 # Définir une graine pour la reproductibilité
 xgbc_pipeline = Pipeline([
    ('preprocessing', preprocesseur),
    ('selection', RFECV(estimator=XGBClassifier(eval_metric='logloss'), step=1, cv=5, scoring='roc_auc')),
    ('classification', XGBClassifier(n_estimators=80, eval_metric="logloss", random_state=set_seed))
 ])
 return xgbc_pipeline

# --- Fonction de Fit et Validation Croisée ---
def executer_evaluation_cv_baselines(les_modeles, metriques, X_train, y_train, X_test, y_test):
    """
    Prend un dictionnaire de modèles/pipelines et une liste de métriques,
    lance la Validation Croisée et retourne un dictionnaire de scores moyens.
    """
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tableau_scores = []
    
    for nom, pipeline in les_modeles.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv_strategy, scoring=metriques)

 
        tableau_scores.append({
            "Modèle": nom,
            "Accuracy": scores['test_accuracy'].mean(),
            "Précision": scores['test_precision'].mean(),
            "Rappel (Recall)": scores['test_recall'].mean(),
            "AUC-ROC": scores['test_roc_auc'].mean(),
            "F1-score": scores['test_f1'].mean()
        })

        pipeline.fit(X_train, y_train)

        # 2. Prédire les probabilités sur le jeu de Test
        y_proba_lr_cv = pipeline.predict_proba(X_test)[:, 1]

        # 3. Calculer le score AUC final sur le Test
        from sklearn.metrics import roc_auc_score
        auc_test_lr_cv = roc_auc_score(y_test, y_proba_lr_cv)

        print(f"✅ AUC de {nom} avec CV sur le TEST : {auc_test_lr_cv:.4f}")
    return tableau_scores

def separation_data_train_test(X:pd.DataFrame, y:pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Sépare les données en ensembles d'entraînement et de test.

    Parameters
    ----------
    X : pd.DataFrame
        Variables explicatives.
    y : pd.Series
        Variable cible.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test, id_test
    """
    # Split des données en train/test
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Extraction sécurisée des IDs pour le fichier de scoring
    id_test = X_test_full['customerID']


    # Suppression de la colonne qui ne sera pas inclue dans le calcul ML
    X_train = X_train_full.drop(columns=['customerID'])
    X_test = X_test_full.drop(columns=['customerID'])

    return X_train, X_test, y_train, y_test, id_test


# Dans src/train.py
import pandas as pd


def extraire_variables_selectionnees(pipeline):
    """Extrait le nom des variables conservées par le RFECV après le preprocessing.

    Retourne une liste de chaînes de caractères.
    """
 
    #Récupération de toutes les variables générées par l'encodage/standardisation
    all_features = pipeline.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    #Récupération du masque booléen du RFECV (True = conservée, False = rejetée)
    rfecv_step = pipeline.named_steps["selection"]

    #Masque des variable selectionnées
    masque_selection = rfecv_step.support_

    # Filtrage pour ne garder que les variables sélectionnées
    features_gardees = all_features[masque_selection]
    #conversion en liste pour un affichage plus lisible
    features_gardees = list(features_gardees)

    print(
        f"RFECV : {len(features_gardees)} variables conservées sur {len(all_features)} initiales.\n"
    )
    return features_gardees, rfecv_step


def afficher_auc_par_nombre_variables_rfecv(rfecv_step):
    # Récupérer les scores moyens d'AUC pour chaque nombre de variables testé
    # (cv_results_ contient la clé 'mean_test_score')
    auc_scores = rfecv_step.cv_results_['mean_test_score']

    print("\nÉvolution du score AUC selon le nombre de variables restantes :")
    for i, score in enumerate(auc_scores, start=1):
        print(f"Top {i} variables -> AUC moyen: {score:.4f}")

    # Le nombre optimal choisi par RFECV
    print(f"\nNombre optimal de variables trouvé : {rfecv_step.n_features_}")


def optimisation_hyperparam_XGB_RSCV(pipeline, X_train, y_train):
    # Définition de la grille pour XGBoost 
    params_grid = {
        'classification__n_estimators': [100, 200, 300],#nombre total d'arbre de décision
        'classification__max_depth': [3, 5, 7],#profondeur des arbres
        'classification__learning_rate': [0.01, 0.05, 0.1],#vitesse (pas) d'apprentissage
        'classification__subsample': [0.8, 1.0]#fraction d'individus pour l'entraînement
    }

    # Lancement de la recherche 
    searchCV_xgb = RandomizedSearchCV(
        pipeline, 
        param_distributions=params_grid, 
        n_iter=10, 
        cv=5, 
        scoring='roc_auc', 
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    searchCV_xgb.fit(X_train, y_train)
    return searchCV_xgb


def extraire_parametres_optimaux(search_xgb):
    """Extrait et affiche proprement les meilleurs hyperparamètres trouvés par

    le RandomizedSearchCV.
    """
    # Récupération du dictionnaire des meilleurs paramètres
    meilleurs_params = search_xgb.best_params_

    print("\n--- ⚙️ Hyperparamètres Optimaux Trouvés ---")
    for cle, valeur in meilleurs_params.items():
        # Nettoyage du nom (ex: 'classification__max_depth' devient 'max_depth')
        nom_propre = cle.split("__")[-1]
        print(f"🔹 {nom_propre} : {valeur}")

    return meilleurs_params
