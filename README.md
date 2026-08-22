# Scoring de churn client

## Introduction
Ce projet concerne la construction d'un modèle de "scoring" capable d'estimer la probabilité de "Churn" pour chaque client, afin de prioriser les actions sur les clients les plus à risque, avec un budget marketing limité.

## Jeu de données 
Le jeu de données fourni est un extrait anonymisé de TelcoWave (un enregistrement par 
client). Il contient des informations démographiques, des services souscrits, des informations contractuelles et de la facturation (tenure, MonthlyCharges, TotalCharges).  
**Variable cible** : Churn (Yes/No) - 'Yes' si le client a résilié dans la dernière période observée, 'No' sinon.  
**Format** : CSV à récupérer dans data/ ou sur Kaggle (Telco 
Customer Churn)
## Structure du projet

```text
Scoring-de-churn-client/
├── README.md
├── data/
│   └── raw/
│       └── telco_customer_churn.csv
├── notebooks/
│   ├── 01_setup_repo_et_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_finetuned_model.ipynb
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── infer.py
│   └── metrics.py
├── models/
│   ├── baseline.joblib
│   └── finetuned.joblib
├── reports/
│   ├── figures/
│   └── model_report.md
├── requirements.txt
└── .gitignore
```

## Installation et Configuration 
Pour exécuter ce projet localement, suivez les étapes suivantes :

```bash
# 1. Cloner le dépôt
git clone https://github.com/ncherfi-lab/Scoring-de-churn-client.git

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```
## Utilisation 
### Option 1 : Mode exploration (notebooks)
Si vous utilisez **VS Code**, ouvrez simplement les fichiers depuis l'explorateur. Sinon, lancez `jupyter lab` dans votre terminal. 

Suivez les notebooks dans l'ordre chronologique pour comprendre la démarche :

1. **`01_setup_repo_et_eda_.ipynb`** : analyse exploratoire des données, visualisation des taux de churn et détection des corrélations clés.
2. **`02_baseline_model.ipynb`** : traitement des valeurs manquantes, encodage des variables catégorielles, utilisation de la régression logistique comme point de référence, génération des métriques (AUC, Recall, Precision, Top ciblage...)
3. **`03_finetuned_model.ipynb`** : Feature Engineering et utilisation de de la RFECV, entraînement du modèle XGBoost avec optimisation des hyperparamètres (RandomizedSearchCV), analyse des courbes de performance (ROC, calibration, lift) et calcul du top k% et le seuil de probabilité pour maximiser les profits.

### Option 2 : Mode Production / Inférence
Pour lancer le pipeline complet et obtenir des prédictions de churn sur de nouvelles données :
```bash
python src/infer.py
```
## Résultats et Performance

Le modèle final retenu est un **XGBoost Classifier**, optimisé par recherche d'hyperparamètres (GridSearchCV).

* **ROC-AUC Score** : `0.846`
* **Précision** : `0.68`

### Facteurs clés du Churn (Feature Importance)
1. **Type de contrat** : les clients avec des contrats au mois (Month-to-month) arrivent à la tête de la liste des clients qui se désabonnent.
2. **Ancienneté du client (Tenure)** : les nouveaux clients ont un risque de départ beaucoup plus élevé.
3. **Ratio évolution facture** : variable crée lors du Feature Engineering indiquant que les variations ou hausses tarifaires sont des éléments déclencheurs du churn.


Les graphiques d'analyse d'importance des variables et la matrice de confusion sont consultables dans le dossier `metrics/` ou directement dans le notebook `models/03_finetuned_model.ipynb`.