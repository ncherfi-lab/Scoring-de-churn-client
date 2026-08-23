# src/train.py
# Manipulation et analyse de données tabulaires
import pandas as pd
# Annotations de type pour les paramètres/retours de fonctions
from typing import Any, Tuple, List
 # Découpage des données, validation croisée stratifiée et recherche aléatoire des meilleurs hyperparamètres
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
# Enchaînement du préprocesseur et du modèle 
from sklearn.pipeline import Pipeline 
 # Sélection récursive de variables avec validation croisée 
from sklearn.feature_selection import RFECV 
# Modèle de classification par boosting de gradient (XGBoost)
from xgboost import XGBClassifier

#########################
# data_split_test_train #
#########################
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

##########################
# entrainer_les_modeles()#
##########################
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

##############################
# assemblage_pipeline_global #
##############################
def assemblage_pipeline_global(preprocesseur: Any, set_seed):
 # Assemblage du Pipeline global
 # Le pipeline applique les transformations, filtre via RFECV, puis entraîne le modèle
 # Définir une graine pour la reproductibilité
 """
    Assemble le pipeline global combinant préprocessing, sélection de variables et classification.

    Parameters
    ----------
    preprocesseur : Any
        Préprocesseur (ColumnTransformer) appliquant les transformations
        aux colonnes catégorielles et numériques.
    set_seed : int
        Graine aléatoire assurant la reproductibilité du modèle.

    Returns
    -------
    Pipeline
        Pipeline scikit-learn enchaînant le préprocessing, la sélection
        récursive de variables (RFECV) et le modèle de classification XGBoost.
 """
 xgbc_pipeline = Pipeline([
    ('preprocessing', preprocesseur),
    ('selection', RFECV(estimator=XGBClassifier(eval_metric='logloss'), step=1, cv=5, scoring='roc_auc')),
    ('classification', XGBClassifier(n_estimators=80, eval_metric="logloss", random_state=set_seed))
 ])
 return xgbc_pipeline

################################
# separation_data_train_test() #
################################
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

####################################
#extraire_variables_selectionnees()#
####################################
def extraire_variables_selectionnees(pipeline: Pipeline) -> Tuple[List[str], RFECV]:
    """
    Extrait le nom des variables conservées par le RFECV après le preprocessing.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline scikit-learn entraîné, contenant une étape "preprocessing"
        et une étape "selection" (RFECV).

    Returns
    -------
    Tuple[List[str], RFECV]
        - Liste des noms des variables sélectionnées par le RFECV
        - Objet RFECV ajusté (contient les attributs comme support_, ranking_...)
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

############################################
# afficher_auc_par_nombre_variables_rfecv()#
############################################
def afficher_auc_par_nombre_variables_rfecv(rfecv_step: RFECV) -> None:
    # Récupérer les scores moyens d'AUC pour chaque nombre de variables testé
    # (cv_results_ contient la clé 'mean_test_score')
    """
    Affiche l'évolution du score AUC moyen en fonction du nombre de variables
    testées par le RFECV, ainsi que le nombre optimal de variables retenu.

    Parameters
    ----------
    rfecv_step : RFECV
        Objet RFECV ajusté (contient les attributs cv_results_ et n_features_).

    Returns
    -------
    None
        La fonction affiche les résultats mais ne retourne rien.
    """
    auc_scores = rfecv_step.cv_results_['mean_test_score']

    print("\nÉvolution du score AUC selon le nombre de variables restantes :")
    for i, score in enumerate(auc_scores, start=1):
        print(f"Top {i} variables -> AUC moyen: {score:.4f}")

    # Le nombre optimal choisi par RFECV
    print(f"\nNombre optimal de variables trouvé : {rfecv_step.n_features_}")

###################################
# optimisation_hyperparam_XGB_RSCV#
###################################
def optimisation_hyperparam_XGB_RSCV(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    # Définition de la grille pour XGBoost 
    """
    Optimise les hyperparamètres du modèle XGBoost via une recherche aléatoire
    avec validation croisée.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline scikit-learn contenant les étapes de preprocessing, sélection
        de variables et classification (XGBoost).
    X_train : pd.DataFrame
        Variables explicatives du jeu d'entraînement.
    y_train : pd.Series
        Variable cible du jeu d'entraînement.

    Returns
    -------
    RandomizedSearchCV
        Objet de recherche ajusté, contenant le meilleur pipeline trouvé
        (best_estimator_), les meilleurs hyperparamètres (best_params_)
        et le meilleur score (best_score_).
    """
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

#################################
# extraire_parametres_optimaux()#
#################################
def extraire_parametres_optimaux(search_xgb: RandomizedSearchCV) -> dict:
    """
    Extrait et affiche proprement les meilleurs hyperparamètres trouvés par
    le RandomizedSearchCV.

    Parameters
    ----------
    search_xgb : RandomizedSearchCV
        Objet de recherche ajusté, contenant les meilleurs hyperparamètres
        trouvés (best_params_).

    Returns
    -------
    dict
        Dictionnaire des meilleurs hyperparamètres, avec les noms préfixés
        par l'étape du pipeline (ex: 'classification__max_depth').
    """
    # Récupération du dictionnaire des meilleurs paramètres
    meilleurs_params = search_xgb.best_params_

    print("\n--- Hyperparamètres Optimaux Trouvés ---")
    for cle, valeur in meilleurs_params.items():
        # Nettoyage du nom (ex: 'classification__max_depth' devient 'max_depth')
        nom_propre = cle.split("__")[-1]
        print(f"🔹 {nom_propre} : {valeur}")

    return meilleurs_params
