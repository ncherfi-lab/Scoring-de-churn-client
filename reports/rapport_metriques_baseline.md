
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
              precision    recall  f1-score   support

   Churn (0)       0.90      0.72      0.80      1035
   Churn (1)       0.50      0.78      0.61       374

    accuracy                           0.74      1409
   macro avg       0.70      0.75      0.71      1409
weighted avg       0.80      0.74      0.75      1409

```

## 4. Conclusion & prochaines étapes
Nous avons utilisé la régression logistique comme point de référence (**baseline**). Le traitement du déséquilibre des classes a été pris en compte (`class_weight='balanced'`), et les métriques obtenues sur le jeu de test sont encourageantes :

* **Stabilité du pouvoir séparateur :** L'AUC atteint **0,8492** pour les données d'entraînement et **0,8413** pour les données de test, confirmant la robustesse du modèle (absence d'overfitting).
* **Validation du Top Ciblage (10 %) :** La simulation métier prouve la capacité de la baseline à prioriser l'urgence. En n'analysant que les **10 % de profils les plus à risque**, le modèle isole une forte densité de vrais positifs (**Précision@10% de 75,71 %**), permettant d'intercepter d'un coup **28,34 %** de l'attrition totale.

Les prochaines itérations (Notebook III) devront inclure :
* Le test d'un modèle non-linéaire plus robuste (**XGBoost**).
* L'optimisation des hyperparamètres (Fine-tuning) pour améliorer le compromis précision/rappel.
* L'intégration de techniques avancées de **Feature Engineering** pour capturer des signaux comportementaux plus complexes.



