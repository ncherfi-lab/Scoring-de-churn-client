
#package pour les tableaux et les DataFrame
import pandas as pd
import numpy as np
#package de visualisation
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
#package sklearn pour les metrics
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    accuracy_score, 
    recall_score, 
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve, 
    roc_auc_score
)
#Pipeline sklearn
from sklearn.pipeline import Pipeline
#sklearn calibration
from sklearn.calibration import calibration_curve

from typing import Tuple
import os
import textwrap

#######################
# model_evaluation()  #
#######################
def model_evaluation(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Évalue les performances d'un modèle sur les jeux train et test.

    Parameters
    ----------
    model : Pipeline
        Modèle entraîné (pipeline scikit-learn).
    X_train : pd.DataFrame
        Variables explicatives du jeu d'entraînement.
    y_train : pd.Series
        Variable cible du jeu d'entraînement.
    X_test : pd.DataFrame
        Variables explicatives du jeu de test.
    y_test : pd.Series
        Variable cible du jeu de test.

    Returns
    -------
    Tuple[float, float]
        AUC sur le jeu d'entraînement et AUC sur le jeu de test.
    """
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # Prédiction des probabilités
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba= model.predict_proba(X_test)[:, 1]

    # Calcul des AUC
    auc_train = roc_auc_score(y_train, y_train_pred_proba)
    auc_test = roc_auc_score(y_test, y_test_pred_proba)
    # Affichez pour chaque modèle les AUC de train et de test
    print(f"AUC : {round(auc_train, 4)} (train)",f"{round(auc_test, 4)} (test)")

    #Calcul Accuracy
    print(f"L'accuracy : {accuracy_score(y_train, y_train_pred)*100:.4f} % (train)")
    print(f"L'accuracy : {accuracy_score(y_test, y_test_pred)*100:.4f} % (test)")

    print("\nRapport de classification (test) :\n")
    print(classification_report(y_test, y_test_pred,target_names=['Churn (0)', 'Churn (1)']))
    # Extraction des rapports complets au format dictionnaire
  

################################
#  recall_and_precison_at_10() #
################################
def recall_and_precison_at_10(
    y_scores:np.ndarray,
    y_test: pd.Series
) ->Tuple:
 """
Calcule le rappel (recall) et la précision (precision) au rang 10,
c'est-à-dire en ne considérant que les 10 éléments les mieux classés
(les 10 plus hauts scores prédits).

Parameters
----------
y_scores : np.ndarray
Scores ou probabilités prédits par le modèle pour chaque échantillon
(plus le score est élevé, plus l'échantillon est considéré comme positif).
y_test : pd.Series
vraies valeurs (0 ou 1) correspondant aux échantillons de test.

Returns
-------
Tuple[float, float]
Un tuple (precision_at_10, recall_at_10) :
- precision_at_10 : proportion de vrais positifs parmi les 10 éléments
 les mieux classés.
- recall_at_10 : proportion de vrais positifs parmi les 10 éléments
 les mieux classés, par rapport au nombre total de positifs réels.
"""
# Convertir les scores en Series Pandas
 scores_series = pd.Series(y_scores)

# Calculer le nombre de lignes du Top 10%
 n_top_10 = int(len(y_test) * 0.10)

# Récupérer les indices des X meilleures valeurs
 top_10_indices = scores_series.nlargest(n_top_10).index

# Créer le vecteur de prédiction
 y_pred_top_10 = np.zeros_like(y_test)
 y_pred_top_10[top_10_indices] = 1

# Calculer les métriques (recall)
 recall_at_10 = recall_score(y_test, y_pred_top_10)
# Calculer les métriques (précision)
 precision_at_10 = precision_score(y_test, y_pred_top_10)

 return recall_at_10, precision_at_10

################
# calcul_AUC() #
################
def calcul_AUC(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple:
 """
    Calcule l'AUC (Area Under the Curve) du modèle sur les jeux
    d'entraînement et de test.

    Parameters
    ----------
    model : Pipeline
        Modèle (ou pipeline scikit-learn) déjà entraîné, utilisé pour
        prédire les probabilités ou scores des échantillons.
    X_train : pd.DataFrame
        Variables explicatives du jeu d'entraînement.
    y_train : pd.Series
        Vraies valeurs du jeu d'entraînement.
    X_test : pd.DataFrame
        Variables explicatives du jeu de test.
    y_test : pd.Series
        Vraies valeurs du jeu de test.

    Returns
    -------
        Tuple[float, float]
        Un tuple (auc_train, auc_test) :
        - auc_train : score AUC calculé sur le jeu d'entraînement.
        - auc_test : score AUC calculé sur le jeu de test.
 """
 # Prédiction des probabilités
 y_train_pred_proba = model.predict_proba(X_train)[:, 1]
 y_test_pred_proba= model.predict_proba(X_test)[:, 1]
 # Calcul des AUC
 auc_train = roc_auc_score(y_train, y_train_pred_proba)
 auc_test = roc_auc_score(y_test, y_test_pred_proba)
 return auc_train, auc_test


#######################################
# generate_metrics_report_basline ()          #
#######################################
def generate_metrics_report_baseline(y_test:pd.Series, y_test_pred:pd.Series) -> None:
    """
    Génère un rapport de performance pour le modèle donné.

    Parameters
    ----------
     y_test : pd.Series
     Vraies valeurs du jeu de test.
     y_test_pred : pd.Series
     Valeurs prédites

    Returns
    --------
     None
         Génère un fichier Markdown contenant le rapport de performance.
    """

    # Chargement du modèle
    # trouver le chemin du fichier 
    # Trouver le chemin absolu du dossier où se trouve metrics.py (src/)
    print("Metrics.py est exécuté depuis le dossier : ", os.getcwd())
    dossier_src = os.path.dirname(os.path.abspath(__file__))
    print(f"Chemin du dossier src : {dossier_src}")

    rapport_classif = classification_report(y_test, y_test_pred,target_names=['Churn (0)', 'Churn (1)'])

    # Génération du Rapport au format Markdown (.md) ---
    markdown_content = '''
# Rapport de Performance - Modèle de référence

## 1. Informations générales
* **Modèle :** Régression Logistique
* **Cible :** Churn (Désabonnement Client)

## 2. Métriques globales

| Métrique | Valeur |
| :--- | :--- |
| **Accuracy (Précision globale)** | 0,7381 |
| **ROC AUC Score** | 0,8413 |

## 3. Rapport de classification
```text
CHIFFRES_CLASSIF
```

## 4. Conclusion & prochaines étapes
Nous avons utilisé la régression logistique comme point de référence (**baseline**). Le traitement du déséquilibre des classes a été pris en compte (`class_weight='balanced'`), et les métriques obtenues sur le jeu de test sont encourageantes :

* **Stabilité du pouvoir séparateur :** L'AUC atteint **0,8492** pour les données d'entraînement et **0,8413** pour les données de test, confirmant la robustesse du modèle (absence d'overfitting).
* **Validation du Top Ciblage (10 %) :** La simulation métier prouve la capacité de la baseline à prioriser l'urgence. En n'analysant que les **10 % de profils les plus à risque**, le modèle isole une forte densité de vrais positifs (**Précision@10% de 75,71 %**), permettant d'intercepter d'un coup **28,34 %** de l'attrition totale.

Les prochaines itérations (Notebook III) devront inclure :
* Le test d'un modèle non-linéaire plus robuste (**XGBoost**).
* L'optimisation des hyperparamètres (Fine-tuning) pour améliorer le compromis précision/rappel.
* L'intégration de techniques avancées de **Feature Engineering** pour capturer des signaux comportementaux plus complexes.



'''

    markdown_final = markdown_content.replace("CHIFFRES_CLASSIF", rapport_classif)
    # Sauvegarde du fichier Markdown

    # Reconstruire le chemin de manière sécurisée vers le dossier reports
    chemin_rapport = os.path.abspath(
        os.path.join(
            dossier_src, "..", "reports", "rapport_metriques_baseline.md"
        )
    )
    print(f"Chemin du rapport : {chemin_rapport}")
    with open(chemin_rapport, "w", encoding="utf-8") as f:
     f.write(textwrap.dedent(markdown_final))
    print("-> Fichier 'rapport_metriques_baseline.md' généré avec succès.")

#########################
# generate_rapport_final#
#########################
def generate_rapport_final() -> None:
    """
    Génère un rapport final enrichi avec les résultats réels du modèle.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Génère un fichier Markdown contenant le rapport final.
    """
    # Contenu du rapport final enrichi avec vos vrais résultats
    markdown_content = '''
<div style="text-align: center; padding-top: 150px; font-family: sans-serif;">
    <h1 style="font-size: 36px; color: #1a3a5f; margin-bottom: 20px;">
        Scoring de churn client
    </h1>
    <h3 style="font-size: 20px; color: #555; font-weight: normal; margin-bottom: 100px;">
        Rapport d'analyse prédictive
    </h3>
    <div style="margin-top: 200px; font-size: 16px; color: #333;">
        <strong>Auteur :</strong> Nawal CHERFI<br>
        <strong>Date :</strong> Août 2026<br>
    </div>
</div>

<!-- Saut de page pour séparer la page de garde du reste -->
<div style="page-break-after: always; break-after: page;"></div>

**Table des matières**
- [Introduction](#introduction)
- [1. Analyse Exploratoire des Données (EDA)](#1-analyse-exploratoire-des-données-eda)
- [2. Démarche et feature engineering](#2-démarche-et-feature-engineering)
- [3. Résultats et comparaison des modèles](#3-résultats-et-comparaison-des-modèles)
- [4. Facteurs de churn et importance par permutation](#4-facteurs-de-churn-et-importance-par-permutation)
- [5. Diagnostic de la calibration et Brier score](#5-diagnostic-de-la-calibration-et-brier-score)
- [6. Optimisation financière du seuil de décision](#6-optimisation-financière-du-seuil-de-décision)
- [7. Stratégie de ciblage et courbe de lift](#7-stratégie-de-ciblage-et-courbe-de-lift)
- [8. Profilage des clients à risque](#8-profilage-des-clients-à-risque)
- [Conclusion et recommandations business](#conclusion-et-recommandations-business)
<!-- Saut de page pour séparer la page de garde du reste -->
<div style="page-break-after: always; break-after: page;"></div>

# Introduction
Dans le secteur des télécommunications et des services numériques, marqué par une forte concurrence, l'attrition des clients est une préoccupation majeure pour les entreprises. Ces dernières déploient des moyens considérables pour fidéliser leur base clientèle et rester compétitives.  
Dans ce projet nous nous intéressons à un opérateur télécom mobile présent en Europe appelé "TelcoWave". La direction « Customer Success » nous confie un enjeu prioritaire : réduire le churn (résiliations) au prochain trimestre. 


# Objectifs du projet et méthodologie
Notre objectif est de construire un modèle de "scoring", capable d'estimer la probabilité de "Churn" pour chaque client, afin de prioriser les actions sur les clients les plus à risque, avec un budget marketing limité.  
Notre travail s'articule autour de deux éléments :
* **Explicabilité des facteurs** : déterminer avec précision les variables qui
déclenchent le départ.
* **Rentabilisation du plan de rétention** : minimiser les coûts marketing en évitant d'offrir des promotions massives à des clients qui n'avaient pas l'intention de se désabonner.  

Notre choix s'est porté sur l'algorithme de machine learning XGBoost, pour sa puissance et sa précision.
Ce rapport présente l'intégralité des résultats, analyses d'interprétabilité et arbitrages stratégiques issus de l'étude prédictive de l'attrition client.

**Jeu de données**  
Le jeu de données fourni est un extrait anonymisé de TelcoWave (un enregistrement par 
client). Il contient des informations démographiques, des services souscrits, des informations contractuelles et de la facturation (tenure, MonthlyCharges, TotalCharges).  
**Variable cible** : Churn (Yes/No) - 'Yes', si le client a résilié dans la dernière période observée. 'No', sinon.  
**Format** : CSV à récupérer dans le répertoire "data/" du projet ou sur Kaggle (Telco Customer Churn).  
Le fichier contient 7043 lignes et 21 colonnes (variables).

<!-- Saut de page pour séparer la page objectif et méthodologie des autres pages -->
<div style="page-break-after: always; break-after: page;"></div>

# 1. Analyse Exploratoire des Données (EDA)
Avant toute modélisation, une analyse exploratoire approfondie a été menée pour identifier les clients à risque.  
Le taux de Churn est plus élevé chez les abonnés ayant souscrit à la **fibre optique**, payant par **chèque électronique**, et ceux qui ont un $$contrat sans engagement renouvelable de mois en mois$$.  
Le graphique ci-dessous par exemple montre que le risque d'attrition est fortement concentré chez les clients ayant la fibre. Le départ de ces clients s'effectue au début de leur abonnement (faible ancienneté). Malgrès ce départ précose, ces clients partent en ayant accumulé des montants importants (boîte orange s'étendant jusqu'à **3 000 €**).
Cela peut s'expliquer par les frais mensuels élevés payés par ces clients.

<figure style="text-align: center;">
  <img src="figures/Distribution_fraistotaux_Internet_Churn.png" alt="Boxplot Churn" width="80%">
  <figcaption><i>Figure 1 : Distribution des frais totaux cumulés selon le type de service Internet et le statut d'attrition.</i></figcaption>
</figure>


# 2. Démarche et feature engineering
Pour capturer les signaux comportementaux des clients, le jeu de données initial a été enrichi par la création de variables spécifiques. 
- **tenure_tranches**: variable découpant la tenure (ancienneté) en tranches par année (6 catégories).  
- **charge_par_mois_tenure** : calcule les frais par mois d'ancienneté ce qui représente les dépenses réelles du client par mois (charge_par_mois_tenure = TotalCharges/tenure).  
- **ratio_evolution_facture** : ratio (MonthlyCharges/ChargeParMoisTenure) d'évolution de la facture. Un ratio supérieur à 1 marque une augmentation.   
- **a_totalcharges_is_missing** : variable binaire indiquant si TotalCharges est manquante ou non (1 si manquant, 0 sinon)  

Afin de garantir la robustesse de cette approche, un protocole rigoureux a été mené :
* **Séparation des variables (Train/Test) :** la séparation 80% Train / 20% Test a été effectuée à la racine pour sanctuariser le jeu de test et éliminer tout risque de *Data Leakage*.
* **Standardisation des données :** nous avons intégré un ColumnTransformer appliquant un OneHotEncoder sur les variables catégorielles et un StandardScaler sur les variables numériques. Les échelles de nos variables étant déjà bien proportionnées et sans écarts extrêmes, une transformation logarithmique n'était pas nécessaire. Le StandardScaler suffit à harmoniser parfaitement les échelles pour nos modèles.
* **Sélection par RFECV (Recursive Feature Elimination with Cross-Validation):** sur les 54 variables générées après encodage, l'algorithme a automatiquement rejeté 16 colonnes redondantes (dont l'indicateur de valeurs manquantes 'a_totalcharges_is_missing'), et a conservé **38 variables prédictives**.<br>
  
Nous avons par la suite utilisé la Régression Logistique comme modèle baseline et amélioré ensuite nos résultats avec un modèle non linéaire (XGBoost).

# 3. Résultats et comparaison des modèles
L'algorithme XGBoost a été utilisé dans un premier temps avec ses hyperparamètres par défaut (à l'exception du paramètre 'n_estimators' fixé à 80) et a abouti aux résultats suivants :  

* **Sur le Train (AUC = 0,9868)** : l'algorithme a appris les données d'entraînement quasiment par cœur, créant des règles ultra-spécifiques pour chaque client.  
* **Sur le Test (AUC = 0,8184)** : face à des données inconnues par l'algorithme, les règles apprises n'ont pas fonctionné. Le score a chuté lourdement à 0,8184.  
  
Une recherche d'hyperparamètres intensive ('RandomizedSearchCV', durée : 30 min 53 sec) a permis de stabiliser le modèle XGBoost.  
Les configurations retenues sont :
* **'max_depth = 3'** : limite la profondeur des arbres pour capturer uniquement les règles macro et éviter l'apprentissage par cœur.
* **'learning_rate = 0,05'** : ralentit la vitesse d'apprentissage pour garantir une convergence prudente et robuste.
* **'n_estimators = 100'** : fixe le nombre d'arbres à un seuil optimal avant l'apparition du surapprentissage.
* **'subsample = 1,0'** : entraîne chaque arbre sur l'intégralité des individus disponibles, le contrôle du surapprentissage étant déjà pleinement assuré par la faible profondeur des arbres ('max_depth').

Comparés à la Régression Logistique (Baseline) sur le jeu de test, les scores d'AUC du modèle XGBoost confirment une excellente robustesse globale :

* **Régression Logistique (Baseline) :** AUC = **0,8413**
* **XGBoost Optimisé (Final) :** AUC = **0,8463**  

L'écart technique de seulement 0,005 (0,5 %) place les deux modèles au même niveau en matière de capacité de classement. La différenciation majeure entre les deux algorithmes s'opérera sur la calibration.
La recherche d'hyperparamètres via 'RandomizedSearchCV' a permis d'aboutir à un gain net de **+2,8 points** sur le jeu de test (0,8184 vs 0,8463).

<figure style="text-align: center;">
  <img src="figures/Courbe ROC - XGBoost_Régression Logistique.png" alt="Courbe ROC" width="80%">
  <figcaption><i>Figure 2 : Courbes ROC</i></figcaption>
</figure>
Dans la zone où le taux de Faux Positifs (FP) se situe entre 10 % et 30 % (axe X), la courbe bleue du XGBoost se détache très légèrement au-dessus de la courbe rouge (Régression logistique). Cela indique que pour un niveau de fausses alertes modéré, le XGBoost intercepte un volume de vrais positifs légèrement supérieur à la Régression Logistique.

# 4. Facteurs de churn et importance par permutation

Nous avons appliqué la méthode de la *Permutation Importance* sur le jeu de test pour identifier les variables qui ont le plus fort pouvoir explicatif sur le départ de nos clients.

<figure style="text-align: center;">
  <!-- Remplacez par le nom réel de votre fichier image -->
  <img src="figures/permutation_importance.png" alt="Top 15 Permutation Importance" width="80%">
  <figcaption><i>Figure 3 : Top 15 des variables explicatives du churn par Permutation Importance.</i></figcaption>
</figure>

L'évaluation de la baisse de l'AUC sur le jeu de test met en lumière le **top 3 des variables clés** décisionnelles :  
* **Type de contrat :** le facteur structurel majeur d'engagement.
* **Tenure (ancienneté) :** l'historique du client.
* **Ratio évolution facture :** la variable financière créée lors du 'Feature Engineering' s'impose avec l'ancienneté et le type du contrat, prouvant que les variations ou hausses tarifaires sont des éléments déclencheus du churn.

# 5. Diagnostic de la calibration et Brier score
Pour valider l'utilisation commerciale directe des probabilités calculées, la calibration a été mesurée sur l'échantillon de test indépendant :
* **Brier score de la Régression Logistique :** 0,1688
* **Brier score XGBoost :** **0,1354** *(Plus proche de 0, donc significativement plus précis)*.  

L'analyse visuelle confirme que la courbe de calibration de **XGBoost** est très proche de la diagonale (qui représente la calibration parfaite). Ses probabilités de risque sont mathématiquement fiables. Au vu de la qualité native du modèle optimisé, **aucun recalibrage a posteriori n'est nécessaire**.

<figure style="text-align: center;">
  <!-- Remplacez par le nom réel de votre fichier image -->
  <img src="figures/courbes_calibration_xgboost_Regressin_Logistique.png" alt="Courbe de Calibration" width="70%">
  <figcaption><i>Figure 4 : Courbes de calibration</i></figcaption>
</figure>
<!-- Saut de page pour séparer la page de garde du reste -->
<div style="page-break-after: always; break-after: page;"></div>

# 6. Optimisation financière du seuil de décision
Sous la contrainte économique d'une campagne de rétention client (coût de l'offre = 15 €, valeur sauvée = 120 €, soit un ratio asymétrique de 1 sur 8), deux stratégies ont été simulées sur le jeu de test :
* **Stratégie A (approche ROI) :** l'optimisation du profit fixe le seuil de déclenchement à **12 % de probabilité de churn**. À ce niveau, le profit net maximal généré atteint **29 565 €**. Augmenter ce seuil fait diminuer le profit car le coût de l'inaction (perdre 120 €) écrase le coût du faux positif (gâcher 15 €).
* **Stratégie B (approche budgétaire) :** le tri des clients par risque décroissant montre que pour capter ce profit maximal de 29 565 €, l'entreprise doit cibler exactement le **Top 60 % des clients les plus instables**. (voir Figure 5).

Le Top 60 % des clients à risque correspond très exactement à la population affichant une probabilité de churn supérieure ou égale à 12 % (probabilité >= 0,12). Les deux stratégies convergent vers la même valeur.


<table style="width: 100%; border: none; border-collapse: collapse; background: transparent;">
  <tr style="border: none; background: transparent;">
    <!-- Colonne Gauche -->
    <td style="width: 50%; border: none; padding: 5px; text-align: center; vertical-align: top;">
      <figure style="margin: 0;">
        <img src="figures/profit_top_K_XGB.png" alt="Courbe des profits XGB" style="width: 100%;">
        <figcaption style="font-size: 0.9em;"><i>Figure 5 : Courbe des profits par Top K% (Modèle XGB)</i></figcaption>
      </figure>
    </td>
    <!-- Colonne Droite -->
    <td style="width: 50%; border: none; padding: 5px; text-align: center; vertical-align: top;">
      <figure style="margin: 0;">
        <img src="figures/Matrice_confusion_XGB.png" alt="matrice de confusion" style="width: 100%;">
        <figcaption style="font-size: 0.9em;"><i>Figure 6 : Matrice de confusion au seuil 12%</i></figcaption>
      </figure>
    </td>
  </tr>
</table>  

Métriques calculées au seuil 12% :  
- Recall : **94,65 %** (354 clients à risque détectés sur 374)  
- Précision : **41,12 %**  
- F1-score : **57,33 %**  

La matrice de confusion (Figure 6) montre que le modèle intercepte près de 95 % des vrais positifs et la simulation financière par Top K% a prouvé que c'est le point d'équilibre parfait pour maximiser le profit net à 29 565 €.  

# 7. Stratégie de ciblage et courbe de lift
* **Au point optimal (Top 60 %)** : le modèle affiche un lift d'environ **1,6**, c'est le point d'équilibre parfait pour optimiser le budget marketing tout en capturant **96 %** de la totalité des résiliations de l'échantillon de test (voir la figure 6).
Le lift de 1,6 signifie qu'en ciblant ce Top 60 % trié par XGBoost, la campagne marketing est 1,6 fois plus efficace qu'un ciblage au hasard.  
Au lieu de retenir seulement 60 % des résiliations (comme le ferait le hasard), nous obtenons :   
60 % de la population x 1,6 (lift) = 96 % de la totalité des résiliations

<figure style="text-align: center;">
  <!-- Remplacez par le nom réel de votre fichier image -->
  <img src="figures/Courbe_lift-XGBoost.png" alt="Courbe de lift" width="70%">
  <figcaption><i>Figure 7 : Courbe de lift du modèle XGBoost</i></figcaption>
</figure>

# 8. Profilage des clients à risque

| Segment Marketing | Tenure (mois) | Facture Mensuelle (€) | Facture Totale (€) | Ratio Évol. Facture | Contrat Mensuel (%) | Contrat 1 An (%) | Contrat 2 Ans (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ne pas cibler (Fidèle)** | 48,2 | 51,63 | 2 822,36 | 1,01 | 7,8 | 34,3 | 57,8 |
| **À cibler (Risque >= 12%)** | 21,5 | 72,02 | 1 815,94 | 1,21 | 84,8 | 13,0 | 2,2 |

Le segment ciblé par la campagne se caractérise par :
* **Le facteur contractuel :** le segment à risque est massivement dominé par les contrats **mensuels (84,8 %)**, tandis que le segment sécurisé (churn = 'No') est caractérisé par des engagements d'un ou deux ans (92,1 %).
* **Le comportement financier :** les clients à risque affichent une ancienneté (tenure) moyenne de 21,5 mois et un 'ratio_evolution_facture' supérieur à 1,0, prouvant qu'ils subissent une instabilité tarifaire sur une période de fidélité encore fragile.
Les frais menseuls moyens sont également elevés (72,02 €).
<!-- Saut de page pour séparer la conclusion du reste -->
<div style="page-break-after: always; break-after: page;"></div>

# Conclusion et recommandations business
Le modèle **XGBoost** est officiellement validé. Il surpasse la Régression Logistique sur la précision des probabilités (Brier score de 0,1354) et offre une rentabilité maximale sécurisée.

**Actions à privilégier :**
* Déclencher l'envoi automatisé du coupon de 15 € dès qu'un client franchit la barre des 12 % de risque calculée par le pipeline, en ciblant prioritairement les profils sans engagement (contrat mensuel) subissant une hausse de tarification récente.  
* Le but de la campagne de rétention ne doit pas seulement être d'offrir une réduction passive. L'objectif doit être de convertir les clients sans engagement à risque vers des contrats avec engagement d'un an, en utilisant le coupon de 15 € comme levier de négociation. Passer un client du contrat mensuel au contrat annuel réduit le risque de churn..

'''
    # Trouver le chemin absolu du dossier où se trouve metrics.py (src/)
    print("Metrics.py est exécuté depuis le dossier : ", os.getcwd())
    dossier_src = os.path.dirname(os.path.abspath(__file__))
    print(f"Chemin du dossier src : {dossier_src}")

    # Écriture finale du rapport

    chemin_rapport = os.path.abspath(
        os.path.join(
            dossier_src, "..", "reports", "rapport_projet_churn_.md"
        )
    )
    print(f"Chemin du rapport : {chemin_rapport}")
    # Reconstruire le chemin de manière sécurisée vers le dossier data/
    chemin_rapport = os.path.abspath(
        os.path.join(
            dossier_src, "..", "reports", "rapport_projet_churn.md"
        )
    )

    with open(chemin_rapport, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print("Le rapport final a été sauvegardé avec succès !")

##############################
# tracer_courbe_calibartion()#
##############################
def tracer_courbe_calibartion(y_true, y_prob, label="Nom du modèle", ax=None, couleur=None, titre=None)-> None:
    """
    Calcule et trace une courbe de calibration sur un graphique unique ou partagé.
    Parameters
    ----------
    y_true : array-like
        Vraies étiquettes (0 ou 1) des échantillons.
    y_prob : array-like
        Probabilités prédites par le modèle pour la classe positive.
    label : str, optional (défaut="Nom du modèle")
        Nom du modèle affiché dans la légende du graphique.
    ax : matplotlib.axes.Axes, optional (défaut=None)
        Axe matplotlib sur lequel tracer la courbe. Si None, un nouvel
        axe (et une nouvelle figure) est créé.
        cela permet de dessiner plusieurs courbes sur le même axe.
    couleur : str, optional (défaut=None)
        Couleur utilisée pour tracer la courbe. Si None, la couleur par
        défaut de matplotlib est utilisée.
    titre : str, optional (défaut=None)
        Titre du graphique. Si None, aucun titre spécifique n'est défini
        (ou un titre par défaut est utilisé).
    affichage_points : bool, optional (défaut=False)
        Si True, affiche également les points individuels de calibration
    en plus de la courbe.

    Returns
    -------
    None
    """
    # Calcul des points
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    # Afficher les valeurs pour chaque point sous forme de tableau
    #if affichage_points:
     # print(f"Valeurs des points de la courbes de {label}:\n")
      #for i, (pred, true) in enumerate(zip(prob_pred, prob_true)):
       #  print(f"Point {i+1} : Prédit (Axe X) = {pred:.1%}, Réel (Axe Y) = {true:.1%}")

    # Gestion de l'axe (création si aucun axe n'est fourni)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        # On trace la ligne de calibration parfaite UNIQUEMENT sur un nouveau graphique
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Parfaite')
    
    # On ajoute la courbe du modèle sur l'axe
    ax.plot(prob_pred, prob_true, "s-", label=label, color=couleur)
    
    ax.set_ylabel("Fraction réelle de Churn")
    ax.set_xlabel("Probabilité moyenne de Churn prédite")
    ax.set_title(titre, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_facecolor('#F0F0F0')
    ax.grid(True, color='white', linestyle='-', linewidth=1.2, zorder=0)
    # ENLEVER LE CADRE (Masquer les 4 bordures)
    for spine in ax.spines.values():
     spine.set_visible(False)
  
    ax.grid(True)
    
    #return ax

###########################
# simuler_profits_top_k() #
###########################
def simuler_profits_top_k(
      y_test: pd.Series, 
      y_pred: pd.Series, 
      cout_offre: float=15, 
      valeur_sauvee: float=120
      )-> Tuple:
    """
    Simule le profit généré en ciblant les clients selon leur score prédit,
    pour différentes valeurs de k (nombre ou proportion de clients ciblés).

    Pour chaque valeur de k, la fonction sélectionne les k clients les mieux
    classés (les plus hauts scores dans y_pred), calcule le coût total des
    offres envoyées ainsi que la valeur sauvée grâce aux vrais positifs
    identifiés parmi ces clients, puis en déduit le profit net.

    Parameters
    ----------
    y_test : array-like
        Vraies valeurs (0 ou 1) des échantillons de test.
    y_pred : array-like
        Scores ou probabilités prédits par le modèle
    cout_offre : float, optional (défaut=15)
    Coût unitaire de l'offre envoyée à chaque client ciblé.
    valeur_sauvee : float, optional (défaut=120)
        Valeur (gain) obtenue pour chaque vrai positif correctement identifié
        et ciblé (par exemple, un client retenu grâce à l'offre).

    Returns
    -------
    Tuple[int, float, list]
     k_optimal : int — nombre d'individus optimal à cibler.
     profit_max : float — profit maximal obtenu.
     profits : list — profits pour chaque valeur de k testée.
    """
    # Créer un DataFrame avec les vraies réponses et les probabilités
    df_topk = pd.DataFrame({
            'realite': y_test,
            'proba': y_pred
    })

    # Trie des clients du PLUS RISQUÉ au MOINS RISQUÉ
    df_topk = df_topk.sort_values(by='proba', ascending=False).reset_index(drop=True)

    # Paramètres financiers
    gain_net_vp = valeur_sauvee - cout_offre
    total_clients = len(df_topk)

    # Simulation pour chaque pourcentage K (de 1% à 100% de la base)
    ks = np.arange(1, 101, 1)
    profits_k = []

    for k in ks:
        # On calcule combien de clients représentent K% de la base
        nb_clients_cibles = int((k / 100) * total_clients)
        
        # On isole ce Top K% de clients les plus risqués
        df_cible = df_topk.iloc[:nb_clients_cibles]
        
        # On compte les Vrais Positifs (ceux qui allaient vraiment partir dans ce groupe)
        vrais_positifs = df_cible['realite'].sum()
        faux_positifs = nb_clients_cibles - vrais_positifs
        
        # Calcul du profit pour ce niveau de budget
        profit = (vrais_positifs * gain_net_vp) - (faux_positifs * cout_offre)
        profits_k.append(profit)

    #Trouver le meilleur K%
    index_max_k = np.argmax(profits_k)
    meilleur_k = ks[index_max_k]
    profit_max_k = profits_k[index_max_k]

    return meilleur_k, profit_max_k, profits_k

##################################
# profilage_clients_churn() #
##################################
def profilage_clients_churn(
        df_analyse: pd.DataFrame, 
        y_pred_proba_XGB: np.ndarray
    )-> pd.DataFrame:
    """
    Établit un profilage des clients selon leur probabilité de churn
    prédite par le modèle XGBoost
    Parameters
    ----------
    df_analyse : pd.DataFrame
        DataFrame contenant les caractéristiques (variables explicatives,
        numériques et catégorielles) des clients à profiler.
    y_pred_proba_XGB : np.ndarray
        Probabilités de churn prédites par le modèle XGBoost pour la classe
        positive (churn)

    Returns
    -------
    pd.DataFrame
        Portrait-robot final des clients à risque de churn
     """
    # Ajout de la colonne "proba_churn"
    df_analyse['proba_churn'] = y_pred_proba_XGB

    # Séparation nette des clients selon votre seuil optimal de 12%
    df_analyse['segment_marketing'] = df_analyse['proba_churn'].apply(
        lambda x: 'À cibler (Risque >= 12%)' if x >= 0.12 else 'Ne pas cibler (Fidèle)'
    )

    # Liste des colonnes clés à analyser
    colonnes_cles = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charge_par_mois_tenure', 'ratio_evolution_facture', 'SeniorCitizen']

    # Calcul du Portrait-Robot moyen
    print("\n--- PROFILAGE-CLIENTS : COMPARAISON DES DEUX SEGMENTS CLIENTS ---")
    
    # le portrait-robot des variables numériques
    portrait_numerique = df_analyse.groupby('segment_marketing')[colonnes_cles].mean().round(3)

    # Le calcul des proportions de contrats (en pourcentage)
    portrait_categoriel = (
        df_analyse.groupby('segment_marketing')['Contract']
        .value_counts(normalize=True)
        .unstack()
        .round(3) * 100
    )

    # Optionnel : renommer les colonnes de contrats pour que le tableau soit parfaitement explicite
    portrait_categoriel = portrait_categoriel.rename(columns={
        'Month-to-month': '%_Contrat_Mois_par_Mois',
        'One year': '%_Contrat_1_An',
        'Two year': '%_Contrat_2_Ans'
    })

    # Concaténation (axis=1 pour coller les tableaux CÔTE À CÔTE)
    portrait_robot_final = pd.concat([portrait_numerique, portrait_categoriel], axis=1)


    return portrait_robot_final

###############################
# tracer_matrices_confusion() #
###############################
def tracer_matrices_confusion(y_verite, y_pred_1, nom_mod_1, y_pred_2=None, nom_mod_2=None):
    """
    Affiche une matrice de confusion seule ou deux matrices côte à côte,
    parfaitement élargies et espacées.
    """
    labels = ['No Churn (0)', 'Churn (1)']
    
    # CAS 1 : Deux matrices côte à côte
    if y_pred_2 is not None and nom_mod_2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5)) 
        
        # Première matrice (à gauche) : Blue
        cm1 = confusion_matrix(y_verite, y_pred_1)
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=labels)
        disp1.plot(cmap=plt.cm.Blues, ax=ax1, values_format='d')
        ax1.set_title(f"Matrice de Confusion\n{nom_mod_1}", fontweight='bold', pad=20, x=0.42)
        
        # Deuxième matrice (à droite) : Red
        cm2 = confusion_matrix(y_verite, y_pred_2)
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=labels)
        disp2.plot(cmap=plt.cm.Reds, ax=ax2, values_format='d')
        ax2.set_title(f"Matrice de Confusion\n{nom_mod_2}", fontweight='bold', pad=20, x=0.42)
        #Sauvegarde paramétrée
        chemin_sauvegarde = f"../reports/figures/matrices_confusion_{nom_mod_1}_{nom_mod_2}.png"
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches="tight")
        print(f"Graphique sauvegardé sous : {chemin_sauvegarde}")
        # AJUSTEMENT : top=0.82 donne de l'air aux titres, wspace=0.55 pousse la matrice verte vers la droite
        plt.subplots_adjust(top=0.82, wspace=0.55) 
        
    # CAS 2 : Une seule matrice
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_verite, y_pred_1)
        # Extraction des valeurs
        vn, fp = cm[0]  # Vrais Négatifs, Faux Positifs
        fn, vp = cm[1]  # Faux Négatifs, Vrais Positifs
        print(f"vn: {vn}, fp: {fp}\n")
        print(f"fn: {fn}, vp: {vp}\n")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        ax.set_title(f"Matrice de Confusion - {nom_mod_1}", fontweight='bold', pad=25, x=0.42)
        plt.subplots_adjust(top=0.85)
        #Sauvegarde paramétrée
        chemin_sauvegarde = f"../reports/figures/matrice_confusion_{nom_mod_1}.png"
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches="tight")
        print(f"Graphique sauvegardé sous : {chemin_sauvegarde}")

    plt.show()


#################################
# tracer_courbes_profit_top_k ()#
#################################
def tracer_courbes_profit_top_k(profits_1, nom_mod_1, meilleur_k_1, profits_2=None, nom_mod_2=None, meilleur_k_2=None):
    """
    Génère la courbe de profit cumulé (Top K%) avec votre syntaxe Plotly Express.
    Affiche 1 modèle ou 2 modèles comparés selon les arguments fournis.
    """
    ks = np.arange(1, 101, 1)
    
    # -----------------------------------
    # CAS 1 : deux modèles (LR +XGBoost)
    # -----------------------------------
    if profits_2 is not None:
        # On initialise la figure avec le premier modèle (en rouge pour la baseline)
        fig = px.line(
            x=ks, y=profits_1, 
            title=f"<b>Comparaison des profits selon le pourcentage K% de clients ciblés</b>",
            labels={'x': 'Pourcentage de la base ciblée (Top K%)', 'y': 'Profit Net Total (€)'}
        )
        fig.update_traces(line=dict(color="blue", width=2), name=nom_mod_1, showlegend=True)
        
        # On ajoute manuellement la deuxième courbe (XGBoost en rouge)
        fig.add_scatter(x=ks, y=profits_2, mode="lines", name=nom_mod_2, line=dict(color="red", width=3))
        # afficher le titre au centre 
        fig.update_layout(title_x=0.5)
        # les lignes verticales d'annotations
        fig.add_vline(x=meilleur_k_1, line_dash="dash", line_color="blue", 
                      annotation_text=f"Top {meilleur_k_1}% ({nom_mod_1})", annotation_position="bottom right")
        fig.add_vline(x=meilleur_k_2, line_dash="dash", line_color="red", 
                      annotation_text=f"Top {meilleur_k_2}% ({nom_mod_2})", annotation_position="top left")
        fig.write_image(f"../reports/figures/profit_top_K_{nom_mod_1}_{nom_mod_2}.png")
    # -----------------------
    # CAS 2 : Un seul modèle 
    # -----------------------
    else:
    
        fig = px.line(
            x=ks, y=profits_1, 
            title=f"<b>{nom_mod_1} : Profit selon le pourcentage K% de clients ciblés</b>",
            labels={'x': 'Pourcentage de la base ciblée (Top K%)', 'y': 'Profit Net Total (€)'}
        )
        # On la met en bleu la courbe XGBoost
        fig.update_traces(line=dict(color="blue", width=3))
        fig.update_layout(title_x=0.5)
       #fig = px.line(x=ks, y=profits_1, title=f"<b>XGBoost : profit selon le pourcentage K% de clients ciblés</b>",
        #              labels={'x': 'Pourcentage de la base ciblée (Top K%)', 'y': 'Profit Net Total (€)'})
        # Votre ligne verticale d'origine
        fig.add_vline(x=meilleur_k_1, line_dash="dash", line_color="blue", 
                      annotation_text=f"Top {meilleur_k_1}%")
        fig.write_image(f"../reports/figures/profit_top_K_{nom_mod_1}.png")
    fig.show()

########################
# calculer_seuil()     #
########################

def calculer_seuil(y_test, y_pred, cout_offre=15, valeur_sauvee=120):

    """
     Calculer le seuil de probabilités pour maximiser les profits
    """
    # Définition des paramètres financiers du problème

    gain_net_vp = valeur_sauvee - cout_offre  # +105€ pour un Vrai Positif

    # Liste de tous les seuils de probabilité XGBoost à tester
    seuils = np.arange(0.01, 1.0, 0.01)
    profits = []

    # Boucle de simulation financière
    for seuil in seuils:
        # Si la probabilité XGBoost >= seuil, on prédit qu'il va partir (1), sinon (0)
        y_pred_seuil = (y_pred >= seuil).astype(int)
        
        # Identification des Vrais Positifs (VP) et Faux Positifs (FP)
        vrais_positifs = np.sum((y_pred_seuil == 1) & (y_test == 1))
        faux_positifs = np.sum((y_pred_seuil == 1) & (y_test == 0))
        
        # Calcul du profit net généré par ce seuil
        profit_total = (vrais_positifs * gain_net_vp) - (faux_positifs * cout_offre)
        profits.append(profit_total)

    # Identification du seuil magique qui maximise l'argent gagné
    index_max = np.argmax(profits)
    seuil_optimal = seuils[index_max]
    profit_max = profits[index_max]
    return seuil_optimal, profit_max, profits

################################
# racer_courbes_profit_seuils()#
################################
def tracer_courbes_profit_seuils(seuil_optimal_1, nom_mod_1, profits_1, seuil_optimal_2=None, nom_mod_2=None, profits_2=None):
    """
    Génère la courbe de profit cumulé (Top K%) avec votre syntaxe Plotly Express.
    Affiche 1 modèle ou 2 modèles comparés selon les arguments fournis.
    """
    seuils = np.arange(0.01, 1.0, 0.01)
    
    # ------------------------------------
    # CAS 1 : Comparaison de deux modèles
    # ------------------------------------
    if seuil_optimal_2 is not None:
        # On initialise la figure avec le premier modèle (en rouge pour la baseline)
        fig = px.line(
            x=seuils, y=profits_1, 
            title="<b>Comparaison des profits selon le seuil de probabilié</b>",
            labels={'x': 'Probabilité', 'y': 'Profit Net Total (€)'}
        )
        fig.update_traces(line=dict(color="blue", width=2), name=nom_mod_1, showlegend=True)
        
        # On ajoute manuellement la deuxième courbe (XGBoost en blue)
        fig.add_scatter(x=seuils, y=profits_2, mode="lines", name=nom_mod_2, line=dict(color="red", width=3))

        # affihcier le litre au centre
        fig.update_layout(title_x=0.5)
        # lignes verticales d'annotations
        fig.add_vline(x=seuil_optimal_1, line_dash="dash", line_color="blue", 
                      annotation_text=f"Seuil {seuil_optimal_1*100:.0f}% ({nom_mod_1})", annotation_position="bottom right")
        fig.add_vline(x=seuil_optimal_2, line_dash="dash", line_color="red", 
                      annotation_text=f"Seuil {seuil_optimal_2*100:.0f}% ({nom_mod_2})", annotation_position="top right")
        fig.write_image(f"../reports/figures/optimisation_seuil_profit_{nom_mod_1}_{nom_mod_2}.png", scale=3)
    # -----------------------
    # CAS 2 : Un seul modèle
    # -----------------------
    else:
        fig = px.line(
            x=seuils, y=profits_1,
            title=f"<b>{nom_mod_1}: optimisation du seuil de décision (Maximum à {seuil_optimal_1:.2f})",
            labels={'x': 'Probabilité', 'y': 'Profit Net Total (€)'}
        )
        fig.update_layout(title_x=0.5)
        fig.add_vline(x=seuil_optimal_1, line_dash="dash", line_color="green", annotation_text=f"Seuil Optimal {seuil_optimal_1:.2f}")
        fig.add_scatter(x=seuils, y=profits_1, mode="lines", name=nom_mod_1, line=dict(color="blue", width=3))
        fig.write_image(f"../reports/figures/optimisation_seuil_profit_{nom_mod_1}.png", scale=3)
    fig.show()

  
##################################
# tracer_courbes_roc()#
##################################
def tracer_courbes_roc(y_test, probas_1, nom_mod_1, couleur_1="blue", probas_2=None, nom_mod_2=None, couleur_2="red"):
    """
    Calcule et affiche la courbe ROC en Plotly.
    Prend en charge l'affichage d'un modèle seul ou de deux modèles superposés.
    Si probas_2 et nom_mod_2 sont renseignées, deux courbes seront représentées.
    """
    # Calcul des points nécessaires pour tracer la courbe ROC du premier modele
    fpr_1, tpr_1, _ = roc_curve(y_test, probas_1)
    auc_1 = roc_auc_score(y_test, probas_1)
    
    # Initialisation du graphique avec la ligne du hasard
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Hasard (AUC = 0.5)', 
                             line=dict(color='gray', dash='dash')))
    
    # --- CAS 1 : Deux modèles superposé ---
    if probas_2 is not None and nom_mod_2 is not None:
        fpr_2, tpr_2, _ = roc_curve(y_test, probas_2)
        auc_2 = roc_auc_score(y_test, probas_2)
        
        # Premier modèle (XGBoost en Bleu par exemple)
        fig.add_trace(go.Scatter(x=fpr_1, y=tpr_1, mode='lines', 
                                 name=f'<b>{nom_mod_1}</b> (AUC: {auc_1:.4f})', 
                                 line=dict(color=couleur_1, width=3)))
        
        # Deuxième modèle (Régression Logistique en Rouge par exemple)
        fig.add_trace(go.Scatter(x=fpr_2, y=tpr_2, mode='lines', 
                                 name=f'{nom_mod_2} (AUC: {auc_2:.4f})', 
                                 line=dict(color=couleur_2, width=2)))
        
        titre = "<b>Comparaison des performances : courbes ROC</b>"
        fig.update_layout(
            title=dict(text=titre, x=0.5, xanchor="center"),
            xaxis_title="Taux de Faux Positifs (1 - Spécificité)",
            yaxis_title="Taux de Vrais Positifs (Rappel)",
            width=850, height=600

        )
        fig.write_image(f"../reports/figures/Courbe ROC - {nom_mod_1}_{nom_mod_2}.png", scale=3)
    # --- CAS 2 : Un seul modèle ---
    else:
        fig.add_trace(go.Scatter(x=fpr_1, y=tpr_1, mode='lines', 
                                 name=f'<b>{nom_mod_1}</b> (AUC: {auc_1:.4f})', 
                                 line=dict(color=couleur_1, width=3)))
        
        titre = f"<b>Performance du Modèle : courbe ROC - {nom_mod_1}</b>"
    
        # Mise en page
        fig.update_layout(
            title=dict(text=titre, x=0.5, xanchor="center"),
            xaxis_title="Taux de Faux Positifs (1 - Spécificité)",
            yaxis_title="Taux de Vrais Positifs (Rappel)",
            width=850, height=600

        )
        fig.write_image(f"../reports/figures/Courbe ROC - {nom_mod_1}.png", scale=3)
    fig.show()


#######################
# tracer_courbe_lift()#
#######################
def tracer_courbe_lift(X_test, y_test, searchCV_xgb):
 """
    Représentation de la courbe de lift
 """
# Trier les vrais y_test selon les probabilités décroissantes de XGBoost
 df_lift = pd.DataFrame({
    'y_true': y_test, 
    'proba': searchCV_xgb.predict_proba(X_test)[:, 1]
 })
 df_lift = df_lift.sort_values(by='proba', ascending=False).reset_index(drop=True)

# Calculer le taux de churn cumulé
 df_lift['cum_churn'] = df_lift['y_true'].cumsum() / df_lift['y_true'].sum()
 df_lift['pop_pct'] = (df_lift.index + 1) / len(df_lift)
 df_lift['lift'] = df_lift['cum_churn'] / df_lift['pop_pct']

 fig = px.line(
            x=df_lift['pop_pct'], y=df_lift['lift'], 
            title="<b>Courbe de Lift - XGBoost</b>",
            labels={'x': 'Proportion de la population ciblée', 'y': 'Lift (Multiplicateur d\'efficacité)'}
        )
 fig.update_layout(title_x=0.5)
 fig.update_traces(line=dict(color="blue", width=2), name="lift", showlegend=True)
 fig.add_hline(y=1, line_dash="dash", line_color="red", 
                      annotation_text="Hasard (Baseline)", annotation_position="top left")
 fig.add_vline(x=0.60, line_dash="dash", line_color="green", 
                      annotation_text="Top 60%", annotation_position="top left")
 fig.write_image(f"../reports/figures/Courbe_lift-XGBoost.png", scale=3)
 fig.show()


#########################
# Matrice de confusion()#
#########################

def tracer_matrices_confusion_seuil(y_verite, y_pred_1, nom_mod_1, seuil):
 """
  Matrice de confision au seuil de probabilité renseigné en paramètres
 """
# Appliquer le seuil financier optimal
 y_pred_optimal = (y_pred_1 >= seuil).astype(int)

 # Calculer la matrice de confusion réelle
 cm = confusion_matrix(y_verite, y_pred_optimal)

 # paramètres d'affichage
 plt.figure(figsize=(4, 3))
 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn (0)', 'Churn (1)'],
            yticklabels=['No Churn (0)', 'Churn (1)'])
 plt.title(f'Matrice de confusion au seuil optimal de 12%-{nom_mod_1}')
 plt.ylabel('Valeurs réelles (observées)')
 plt.xlabel('Valeurs prédictives')
 plt.savefig(f"../reports/figures/Matrice_confusion_XGB.png", dpi=300, bbox_inches='tight')
 plt.show()

