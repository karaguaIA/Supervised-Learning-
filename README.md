# Supervised Learning — Classification

Dépôt personnel de travaux pratiques (TP) sur le thème **Apprentissage Supervisé — Classification**, réalisés dans le cadre de mon apprentissage du Machine Learning. Chaque TP est un notebook Jupyter/Google Colab qui va de la préparation des données jusqu'à l'évaluation de modèles de classification, avec une explication du fonctionnement mathématique de chaque algorithme.

## Structure du dépôt

```
Classification/
├── Doc 1/
│   ├── Car_Data.csv                              # Données de véhicules scrapées sur expat-dakar.com
│   ├── Clas_KNN_on_Car_Data.ipynb                # Version à compléter (exercice)
│   └── Copie de Clas_KNN_on_Car_Data.ipynb       # Version corrigée avec explications mathématiques
└── Clas_Log_Reg_SVM_Decision_Tree_uncomplet.ipynb # Régression Logistique, SVM et Decision Tree sur données bancaires
```

## TP 1 — K plus proches voisins (K-NN)

**Fichier :** `Classification/Doc 1/Copie de Clas_KNN_on_Car_Data.ipynb`
**Données :** `Car_Data.csv` — annonces de véhicules à Dakar (marque, année, boîte de vitesse, quartier, prix).
**Objectif :** prédire si une voiture est **venante** (neuve) ou **d'occasion**.

Étapes couvertes : nettoyage des variables (adresse → quartier), encodage des variables catégorielles, normalisation (`MinMaxScaler`), recherche du $k$ optimal par validation croisée (`GridSearchCV`), prédiction et évaluation (accuracy).

Le notebook explique la mécanique du K-NN : calcul de distance (euclidienne / Manhattan / Minkowski), règle du vote majoritaire (et vote pondéré), rôle de l'hyperparamètre $k$, et complexité algorithmique.

## TP 2 — Régression Logistique, SVM & Decision Tree

**Fichier :** `Classification/Clas_Log_Reg_SVM_Decision_Tree_uncomplet.ipynb`
**Données :** `Bank_Data.csv` (jeu [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) de l'UCI repository) — campagnes de marketing téléphonique d'une banque portugaise.
**Objectif :** prédire si un client souscrira un dépôt à terme (`y`).

Étapes couvertes : exploration (countplots, boxplots, pairplot), traitement des valeurs aberrantes (IQR), encodage et normalisation (`StandardScaler`), split train/val/test, recherche d'hyperparamètres par `GridSearchCV`, puis entraînement, évaluation (accuracy, F1-score, matrice de confusion) et comparaison des trois modèles.

Le notebook explique le fonctionnement mathématique de chaque algorithme :
- **Régression Logistique** : combinaison linéaire, fonction sigmoïde, frontière de décision, log-loss, régularisation L1/L2.
- **SVM** : hyperplan à marge maximale, marge souple (`C`), astuce du noyau (kernel RBF).
- **Decision Tree** : partitionnement récursif, indices d'impureté (Gini, entropie), critère de gain, profondeur maximale.

## Prérequis

Les notebooks sont conçus pour **Google Colab** (montage de Google Drive pour charger les CSV). Pour une exécution en local, remplacer les cellules `drive.mount(...)` et adapter les chemins `pd.read_csv(...)` vers les fichiers du dossier `Doc 1/`, puis installer :

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

## Bibliothèques utilisées

- `pandas`, `numpy` — manipulation de données
- `matplotlib`, `seaborn`, `plotly` — visualisation
- `scikit-learn` — préparation des données (`LabelEncoder`, `MinMaxScaler`, `StandardScaler`), modélisation (`KNeighborsClassifier`, `LogisticRegression`, `SVC`, `DecisionTreeClassifier`), sélection d'hyperparamètres (`GridSearchCV`) et métriques (`accuracy_score`, `f1_score`, `confusion_matrix`)
