
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  
from sklearn.metrics import roc_curve, roc_auc_score
import plotly.graph_objects as go

#from matplotlib.path import Path
from typing import Any, Tuple

import joblib
import sys
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
 # Prédiction des probabilités
 y_train_pred_proba = model.predict_proba(X_train)[:, 1]
 y_test_pred_proba= model.predict_proba(X_test)[:, 1]
 # Calcul des AUC
 auc_train = roc_auc_score(y_train, y_train_pred_proba)
 auc_test = roc_auc_score(y_test, y_test_pred_proba)
 return auc_train, auc_test


#######################################
# generate_metrics_report ()          #
#######################################
def generate_metrics_report(y_test, y_test_pred) -> None:
    """
    Génère un rapport de performance pour le modèle donné.

    # Parameters
    # ----------
    # None
    #
    # Returns
    # -------
    # None
    #     Génère un fichier Markdown contenant le rapport de performance.
    # """

    # Chargement du modèle
    # trouver le chemin du fichier 
    # Trouver le chemin absolu du dossier où se trouve metrics.py (src/)
    print("Metrics.py est exécuté depuis le dossier : ", os.getcwd())
    dossier_src = os.path.dirname(os.path.abspath(__file__))
    print(f"Chemin du dossier src : {dossier_src}")

    rapport_classif = classification_report(y_test, y_test_pred,target_names=['Churn (0)', 'Churn (1)'])

    # Génération du Rapport au format Markdown (.md) ---
    markdown_content = '''
# Rapport de Performance - Modèle Baseline

## 1. Informations Générales
* **Modèle :** Régression Logistique (Baseline)
* **Cible :** Churn (Désabonnement Client)

## 2. Métriques Globales

| Métrique | Valeur |
| :--- | :--- |
| **Accuracy (Précision globale)** | 0.7381 |
| **ROC AUC Score** | 0.8413 |

## 3. Rapport de Classification Élargi
```text
CHIFFRES_CLASSIF
```

## 4. Conclusion & Prochaines Étapes
Nous avons utilisé la régression logistique comme point de référence (**baseline**). Le traitement du déséquilibre des classes a été pris en compte (`class_weight='balanced'`), et les métriques obtenues sur le jeu de test sont encourageantes :

* **Stabilité du pouvoir séparateur :** L'AUC atteint **0.8492** pour les données d'entraînement et **0.8413** pour les données de test, confirmant la robustesse du modèle (absence d'overfitting).
* **Validation du Top Ciblage (10 %) :** La simulation métier prouve la capacité de la baseline à prioriser l'urgence. En n'analysant que les **10 % de profils les plus à risque**, le modèle isole une forte densité de vrais positifs (**Précision@10% de 75.71 %**), permettant d'intercepter d'un coup **28.34 %** de l'attrition totale.

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
            dossier_src, "..", "reports", "rapport_metriques.md"
        )
    )
    print(f"Chemin du rapport : {chemin_rapport}")
    with open(chemin_rapport, "w", encoding="utf-8") as f:
     f.write(textwrap.dedent(markdown_final))
    print("-> Fichier 'rapport_metriques.md' généré avec succès.")

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
# Rapport de synthèse final : optimisation et évaluation du modèle de Churn

Ce rapport présente l'intégralité des résultats, analyses d'interprétabilité et arbitrages stratégiques issus de l'étude prédictive de l'attrition client.

## 1. Démarche et Feature Engineering
Pour capturer les signaux comportementaux des clients, le jeu de données initial a été enrichi par la création de variables spécifiques. Afin de garantir la robustesse de cette approche, un protocole rigoureux a été mené :
* **Cloisonnement strict (Train/Test) :** la séparation 80% Train / 20% Test a été effectuée à la racine pour sanctuariser le jeu de test et éliminer tout risque de *Data Leakage*.
* **Standardisation des données :** nous avons intégré un ColumnTransformer appliquant un OneHotEncoder sur les variables catégorielles et un StandardScaler sur les variables numériques. Les échelles de nos variables étant déjà bien proportionnées et sans écarts extrêmes, une transformation logarithmique n'était pas nécessaire. Le StandardScaler suffit à harmoniser parfaitement les échelles pour nos modèles.
* **Sélection par RFECV :** sur les 54 variables générées après encodage, l'algorithme a automatiquement rejeté 16 colonnes redondantes (dont l'indicateur de valeurs manquantes `a_totalcharges_is_missing`), verrouillant le modèle sur un cœur de **38 variables hautement prédictives**.

## 2. Match des Champions : Résultats sur le Jeu de Test
Une recherche d'hyperparamètres intensive (`RandomizedSearchCV`, durée : 30min 53sec) a permis de stabiliser le modèle XGBoost. Confronté à la Régression Logistique (Baseline) sur le jeu de test indépendant, les scores d'AUC confirment une excellente robustesse globale :
* **Régression Logistique (Baseline) :** AUC = **0.8413**
* **XGBoost Optimisé (Final) :** AUC = **0.8463**

### Hyperparamètres optimisés par RandomizedSearchCV
La recherche d'hyperparamètres a permis de brider la puissance de calcul de XGBoost pour éliminer le surapprentissage initial (gain de +2.8 points sur le jeu de test) :

| Hyperparamètre | Valeur Optimale | Rôle Métier / Technique |
| :--- | :---: | :--- |
| `max_depth` | **3** | lmite la profondeur des arbres pour capturer uniquement les règles macro. |
| `learning_rate` | **0.05** | ralentit l'apprentissage pour une correction prudente et robuste des erreurs. |
| `n_estimators` | **100** | fixe le nombre d'arbres à un seuil optimal avant l'apparition de l'overfitting. |
| `subsample` | **1.0** | utilise l'intégralité des individus de la poche d'entraînement à chaque étape. |

L'écart technique de seulement 0.005 (0.5 %) place les deux modèles au même niveau d'excellence en matière de capacité intrinsèque de classement.

## 3. Diagnostic de la Calibration et Brier Score
Pour valider l'utilisation commerciale directe des probabilités calculées, la calibration a été mesurée sur l'échantillon de test indépendant :
* **Brier Score Baseline :** 0.1688
* **Brier Score XGBoost :** **0.1354** *(Plus proche de 0, donc significativement plus précis)*.  

L'analyse visuelle confirme que la courbe de calibration de **XGBoost** est très proche de la diagonale (qui représente la calibration parfaite). Ses probabilités de risque sont mathématiquement fiables. Au vu de la qualité native du modèle optimisé, **aucun recalibrage post-processing n'est nécessaire**.

## 4. Optimisation Financière du Seuil de Décision
Sous la contrainte économique d'une campagne de rétention (Coût de l'offre = 15 €, Valeur sauvée = 120 €, soit un ratio asymétrique fort de 1 à 8), deux stratégies ont été simulées sur le jeu de test :
* **Stratégie A (Approche ROI) :** l'optimisation du profit fixe le seuil de déclenchement à **12 % de probabilité de churn**. À ce niveau, le profit net maximal généré atteint **29.56 k€**. Augmenter ce seuil fait diminuer le profit car le coût de l'inaction (perdre 120 €) écrase le coût du faux positif (gâcher 15 €).
* **Stratégie B (Approche Budgétaire) :** le tri des clients par risque décroissant montre que pour capter ce profit maximal de 29.56 k€, l'entreprise doit cibler exactement le **Top 60 % des clients les plus instables**.

**Le Lien Logique :** le Top 60 % des clients les plus dangereux correspond très exactement à la population affichant une probabilité de churn supérieure ou égale à 12 % (`proba >= 0.12`). Les deux stratégies convergent vers le même sommet économique.

## 5. Interprétabilité et Portrait-Robot
### Importance par Permutation (Permutation Importance)
L'évaluation de la perte d'AUC sur le jeu de test mettre en lumière le **Top 3 des variables clés** dictant le comportement de l'algorithme :
1. **Type de Contrat :** le facteur structurel majeur d'engagement.
2. **Tenure (Ancienneté) :** l'historique du client.
3. **Ratio Évolution Facture :** la variable financière créée lors du Feature Engineering s'impose avec l'ancienneté et le type du contrat, prouvant que les variations ou hausses tarifaires récentes sont le déclencheur direct du churn.

### Portrait-Robot du Segment à Risque (Top 60% / Proba >= 12%)
La confrontation des variables numériques et catégorielles valide les priorités du modèle XGBoost. Le segment ciblé par la campagne se caractérise par :
* **Le facteur contractuel :** le segment à risque est massivement dominé par les contrats **Mois par Mois (84.8 %)**, tandis que le segment fidèle est sécurisé par des engagements d'un ou deux ans (92.1 %).
* **Le comportement financier :** les clients à risque affichent une tenure (ancienneté) moyenne nettement plus faible et un `ratio_evolution_facture` supérieur à 1.0, prouvant qu'ils subissent une instabilité tarifaire sur une période de fidélité encore fragile.

## 6. Conclusion et Recommandations Business
Le modèle **XGBoost** est officiellement validé. Il surpasse la baseline sur la précision des probabilités (Brier Score de 0.1354) et offre une rentabilité maximale sécurisée.

**Actions à privilégier :**
* Déclencher l'envoi automatisé du coupon de 15 € dès qu'un client franchit la barre des **12 % de risque** calculée par le pipeline, en ciblant prioritairement les profils sans engagement (contrat mensuel) subissant une hausse de tarification récente.
* Le but de la campagne de rétention ne doit pas seulement être d'offrir une réduction passive. L'objectif doit être de **convertir les clients Month-to-month à risque vers des contrats avec engagement d'un an**, en utilisant le coupon de 15 € comme levier de négociation. Passer un client du contrat mensuel au contrat annuel détruit statistiquement ses chances de churner sans les éliminer.

'''
    # Trouver le chemin absolu du dossier où se trouve metrics.py (src/)
    print("Metrics.py est exécuté depuis le dossier : ", os.getcwd())
    dossier_src = os.path.dirname(os.path.abspath(__file__))
    print(f"Chemin du dossier src : {dossier_src}")

    # Écriture finale du rapport

    chemin_rapport = os.path.abspath(
        os.path.join(
            dossier_src, "..", "reports", "rapport_metriques_final.md"
        )
    )
    print(f"Chemin du rapport : {chemin_rapport}")
    # Reconstruire le chemin de manière sécurisée vers le dossier data/
    chemin_rapport = os.path.abspath(
        os.path.join(
            dossier_src, "..", "reports", "rapport_metriques_final.md"
        )
    )

    with open(chemin_rapport, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print("Le rapport final intégrant les 3 consignes et vos vrais chiffres a été sauvegardé avec succès !")

##############################
# tracer_courbe_calibartion()#
##############################
def tracer_courbe_calibartion(y_true, y_prob, label="Nom du modèle", ax=None, couleur=None, titre=None, affichage_points=False):
    """
    Calcule et trace une courbe de calibration sur un graphique unique ou partagé.
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
    
    return ax

###########################
# simuler_profits_top_k() #
###########################
def simuler_profits_top_k(y_test, y_pred, cout_offre=15, valeur_sauvee=120):
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
# portrait_robot_clients_churn() #
##################################
def portrait_robot_clients_churn(df_analyse, y_pred_proba_XGB):
    # Ajout de la colonne "proba_churn"
    df_analyse['proba_churn'] = y_pred_proba_XGB

    # Séparation nette des clients selon votre seuil optimal de 12%
    df_analyse['segment_marketing'] = df_analyse['proba_churn'].apply(
        lambda x: 'À cibler (Risque >= 12%)' if x >= 0.12 else 'Ne pas cibler (Fidèle)'
    )

    # Liste des colonnes clés à analyser
    colonnes_cles = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charge_par_mois_tenure', 'ratio_evolution_facture', 'SeniorCitizen']

    # Calcul du Portrait-Robot moyen
    print("\n--- PORTRAIT-ROBOT : COMPARAISON DES DEUX SEGMENTS CLIENTS ---")
    
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
    # 1. Définition des paramètres financiers du problème

    gain_net_vp = valeur_sauvee - cout_offre  # +105€ pour un Vrai Positif

    # 2. Liste de tous les seuils de probabilité XGBoost à tester
    seuils = np.arange(0.01, 1.0, 0.01)
    profits = []

    # 3. Boucle de simulation financière
    for seuil in seuils:
        # Si la probabilité XGBoost >= seuil, on prédit qu'il va partir (1), sinon (0)
        y_pred_seuil = (y_pred >= seuil).astype(int)
        
        # Identification des Vrais Positifs (VP) et Faux Positifs (FP)
        vrais_positifs = np.sum((y_pred_seuil == 1) & (y_test == 1))
        faux_positifs = np.sum((y_pred_seuil == 1) & (y_test == 0))
        
        # Calcul du profit net généré par ce seuil
        profit_total = (vrais_positifs * gain_net_vp) - (faux_positifs * cout_offre)
        profits.append(profit_total)

    # 4. Identification du seuil magique qui maximise l'argent gagné
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
                      annotation_text=f"Seuil {seuil_optimal_1*100:.2f}% ({nom_mod_1})", annotation_position="bottom right")
        fig.add_vline(x=seuil_optimal_2, line_dash="dash", line_color="red", 
                      annotation_text=f"Seuil {seuil_optimal_2*100:.2f}% ({nom_mod_2})", annotation_position="top right")
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
        fig.add_vline(x=seuil_optimal_1, line_dash="dash", line_color="green", annotation_text="Seuil Optimal")
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
    """
    # 1. Calcul des points nécessaires pour tracer la courbe ROC du premier modele
    fpr_1, tpr_1, _ = roc_curve(y_test, probas_1)
    auc_1 = roc_auc_score(y_test, probas_1)
    
    # 2. Initialisation du graphique avec la ligne du hasard
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Hasard (AUC = 0.5)', 
                             line=dict(color='gray', dash='dash')))
    
    # --- CAS 1 : Deux modèles superposés (Partie 3 - Comparaison) ---
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
    # --- CAS 2 : Un seul modèle (Partie 1 ou Partie 2 seul) ---
    else:
        fig.add_trace(go.Scatter(x=fpr_1, y=tpr_1, mode='lines', 
                                 name=f'<b>{nom_mod_1}</b> (AUC: {auc_1:.4f})', 
                                 line=dict(color=couleur_1, width=3)))
        
        titre = f"<b>Performance du Modèle : courbe ROC - {nom_mod_1}</b>"
    
        # 3. Mise en page professionnelle
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


