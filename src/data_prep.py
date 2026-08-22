#Definition des fonctions de preprocessing

#Import des librairies 
import pandas as pd
import numpy as np
import os

from typing import List, Tuple
#pour le preprosessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from pathlib import Path

######################
# Import des donnees #
######################
#    
def load_data(nom_du_fichier:str)-> pd.DataFrame: 
    """
    Charge le jeu de données.

    Parameters
    ----------
    nom_du_fichier : str, nom
        du fichier CSV à charger.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données brutes.
    """
    print(nom_du_fichier)
    # 1. Trouver le chemin absolu de la racine du projet de manière élégante
    # Si train.py est dans src/, .parent remonte directement à la racine du projet
    racine_projet = Path(__file__).resolve().parent.parent

    # 2. Reconstruire le chemin de manière ultra-lisible avec l'opérateur "/"
    # Plus besoin d'imbriquer des os.path.join() illisibles !
    chemin_csv = racine_projet / "data" / nom_du_fichier

     # 3. --- BLOC DE SÉCURITÉ : Vérification visuelle ---
    if not chemin_csv.exists():
       raise FileNotFoundError(
        f"❌ Le fichier CSV est introuvable à cet emplacement précis : {chemin_csv}\n"
        f"Vérifiez l'orthographe exacte du nom du fichier dans votre dossier 'data'."
    )
    # 3. Lecture sécurisée
    #print(f"Chargement du fichier depuis : {chemin_csv}")
    return pd.read_csv(chemin_csv)

#######################################
# Identification du type des colonnes #
#######################################
def get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identifie les colonnes catégorielles et numériques d'un DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame contenant les variables explicatives.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        - Liste des colonnes catégorielles
        - Liste des colonnes numériques
        - Liste de toutes les colonnes
    """
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    all_cols = X.columns.tolist()
    return cat_cols, num_cols, all_cols

############################
# Séparation des variables #
############################
def features_target_separation(df:pd.DataFrame, col_target:str, col_change_type: str|None = None)-> Tuple[pd.DataFrame, pd.Series]:
    """
    Separe les variables explicatives (features) de la variable cible (target) dans un DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.
    col_target : str
        Nom de la colonne contenant la variable cible.
    col_change_type : str, optional
        Nom de la colonne dont le type doit être changé en numérique.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Les variables explicatives et la variable cible.
    """

    # Suppression de la colonne qui ne sera pas inclue dans le calcul ML
    #df.drop([col_to_drop], axis=1, inplace=True)

    #changement du type de la variable object->float
    if col_change_type is not None:
        df[col_change_type] = pd.to_numeric(df[col_change_type], errors='coerce')

    # vérifier si la colonne cible ('Churn') est présente dans le DataFrame
    if col_target in df.columns:
       # Séparation des variables explicatives de la variable cible
       X = df.drop([col_target], axis=1)
       y = df[col_target]
       # Encodage de la variable cible en binaire (0/1)
       y = y.map({"No": 0, "Yes": 1})
    else:
        # pour (infer.py), la cible n'existe pas
        X = df.copy()
        y = None # On renvoie None à la place de y      

    return X,y


# src/data_prep.py


def data_preprocessing(X:pd.DataFrame) -> ColumnTransformer:
  # Séparation des colonnes numériques et catégorielles
  num_cols = X.select_dtypes(include=np.number).columns
  cat_cols = X.select_dtypes(exclude=np.number).columns

  print(f"Colonnes numériques : {num_cols}")
  print(f"Colonnes catégorielles : {cat_cols}")

  #Pipeline pour le traitement des variables numériques
  pipeline_num = Pipeline(
          [
              ("imputer_num", SimpleImputer(strategy="median")),
              ("scaler", StandardScaler()),
          ]
      )
  #Pipeline pour le traitement des variables catégorielles
  pipeline_cat = Pipeline(
          [
              ("imputer_cat", SimpleImputer(strategy="constant", fill_value="missing")),
              ("encoder",OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
          ]
      )

   # Combinaison des deux pipelines
  preprocesseur = ColumnTransformer(
        transformers=[
            ("var_num", pipeline_num, num_cols),
            ("var_cat", pipeline_cat, cat_cols),
        ]
    )
  return preprocesseur

#def Feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
def Feature_engineering(df: pd.DataFrame)->Tuple[pd.DataFrame, pd.Series]:
    """
    Effectue l'ingénierie des fonctionnalités sur les données.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Les données features et la variable cible.
    """
 
   # Création de tranches d'années (ex: 0-12 mois, 12-24 mois, 24-36 mois, et plus)
    max_tenure = df['tenure'].max()
    tranche_finale= max_tenure if max_tenure > 60 else 72 #ajouter une sécurité pour éviter d'avoir des doublons si la max=60
    bins = [0, 12, 24, 36, 48, 60, tranche_finale]
    labels = ['tenure_0_1yr', 'tenure_1_2yr', 'tenure_2_3yr', 'tenure_3_4yr', "tenure_4_5yr", "tenure_5_6yr"]

    # Nouvelle variable catégorielle
    df['tenure_tranches'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)

    # Conversion de la colonne 'TotalCharges' en numérique, en remplaçant les valeurs non convertibles par NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Calcul de la valeur financière moyenne par mois d'ancienneté 
    #ce que le client dépense hors forfait par mois
    df['charge_par_mois_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
 
    #Ratio d'évolution de la facture (Proche de 1 = stable, supérieur à 1 = augmentation récente)
    df['ratio_evolution_facture'] = df['MonthlyCharges'] / (df['charge_par_mois_tenure'] + 1)

    #Ajout d'une variable binaire indiquant si TotalCharges est manquant ou non (1 si manquant, 0 sinon)
    df['a_totalcharges_is_missing'] = df['TotalCharges'].isnull().astype(int)

    print("\nDimensions du DataFrame après Feature Engineering :")
    print(f"Nombre de lignes: {df.shape[0]}, Nombre de colonnes: {df.shape[1]}")

    return features_target_separation(df, col_target='Churn', col_change_type=None)

    # Ensuite, vous l'encodez en variables indicatrices (Dummy variables) avant le RFECV

    #df = pd.get_dummies(df, columns=['tenure_tranches'], drop_first=False)
    #return X, y

