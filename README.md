# 🎓 Projet Data Science IGA - Prédiction des Retours E-commerce

## 📋 Réponse aux Exigences du Cahier des Charges

### I. Objectif du Projet ✅

**Problématique métier définie** : 
- **Contexte** : Le commerce électronique connaît une croissance exponentielle, mais les retours de produits représentent un défi majeur pour les entreprises. Les retours génèrent des coûts logistiques, affectent la satisfaction client et impactent la rentabilité.
- **Variable cible** : `will_return` (binaire) : 1 si le produit sera retourné, 0 sinon
- **Intérêt du Machine Learning** : Réduction des coûts logistiques, optimisation du stockage, amélioration de l'expérience client, prise de décision proactive

### II. Données & Pré-processing ✅

**Exploration des données (EDA)** :
- ✅ Analyse exploratoire complète dans `01_exploration_donnees.ipynb`
- ✅ Visualisation des distributions et corrélations
- ✅ Identification des valeurs manquantes et aberrantes

**Gestion des données** :
- ✅ Gestion des valeurs manquantes (imputation médiane/mode)
- ✅ Gestion des valeurs aberrantes (méthode IQR)
- ✅ Encodage des variables catégorielles (OneHotEncoder)
- ✅ Standardisation des variables numériques (StandardScaler)
- ✅ Séparation des données en train/validation/test (70%/15%/15%)

### III. Modélisation ✅

**Trois algorithmes implémentés** :
1. **Régression Logistique** : Algorithme linéaire de base
2. **Random Forest** : Algorithme d'ensemble robuste
3. **SVM (Support Vector Machine)** : Algorithme à noyau puissant

**Comparaison des performances** :
- ✅ Métriques complètes : Accuracy, Precision, Recall, F1-Score, ROC AUC
- ✅ Visualisation comparative des performances
- ✅ Matrices de confusion pour chaque modèle

**Justification du modèle final** :
- ✅ Sélection basée sur les meilleures performances ROC AUC
- ✅ Analyse des compromis biais-variance
- ✅ Interprétabilité et complexité considérées

### IV. Tuning & Pipelines ✅

**Pipeline scikit-learn rigoureux** :
- ✅ Pipeline complet avec prétraitement et classification
- ✅ Imputation + Encodage + Standardisation intégrés
- ✅ Reproductibilité garantie

**Optimisation avec GridSearchCV** :
- ✅ Grilles d'hyperparamètres définies pour chaque algorithme
- ✅ Validation croisée 5-fold intégrée
- ✅ Optimisation basée sur la métrique ROC AUC

### V. Déploiement ✅

**Application Flask fonctionnelle** :
- ✅ Interface web accessible via `app.py`
- ✅ Formulaire de saisie des données utilisateur
- ✅ Prédictions en temps réel avec probabilités
- ✅ Affichage des résultats en français

**Fonctionnalités** :
- ✅ Saisie : Quantité, Prix, Remise, Port, Catégorie, Canal, Paiement, Pays
- ✅ Prédiction : Probabilité de retour avec niveau de confiance
- ✅ Interface : Design moderne avec Tailwind CSS

### VI. Démonstration & Valorisation ✅

**Préparation vidéo** :
- ✅ Notebook principal `project_iga.ipynb` structuré pour démonstration
- ✅ Résultats visuels et métriques prêts à présenter
- ✅ Pipeline complet de bout en bout

## 🏗️ Structure du Projet

```
ecommerce_return_prediction/
├── 📓 notebooks/                    # Notebooks d'analyse
│   ├── 01_exploration_donnees.ipynb    # EDA complet
│   ├── 02_pretraitement_donnees.ipynb   # Prétraitement
│   ├── 03_modelisation.ipynb            # Modélisation
│   └── 04_deploiement.ipynb            # Déploiement
├── 🤖 models/                      # Modèles entraînés
├── 📊 data/                        # Données brutes et traitées
│   └── raw/
│       └── business.retailsales.csv   # Dataset principal
├── 🌐 app.py                       # Application Flask
├── 🎨 templates/
│   └── index.html                 # Interface web
├── 📋 project_iga.ipynb           # Notebook principal IGA
└── 📦 requirements.txt             # Dépendances
```

## 🚀 Lancement du Projet

### 1. Installation
```bash
# Activation environnement virtuel
source ../.venv/bin/activate

# Installation dépendances
pip install flask pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 2. Entraînement
```bash
# Lancer le notebook principal
jupyter notebook project_iga.ipynb
```

### 3. Déploiement
```bash
# Lancer l'application web
python app.py
# Visiter : http://localhost:5000
```

## 📈 Résultats Obtenus

### Performance des Modèles
- **Régression Logistique** : Baseline interprétable
- **Random Forest** : Meilleure performance globale
- **SVM** : Bon compromis précision/complexité

### Pipeline Complet
1. **Chargement** : Dataset `business.retailsales.csv`
2. **Nettoyage** : Gestion valeurs manquantes/aberrantes
3. **Prétraitement** : Encodage + Standardisation
4. **Modélisation** : 3 algorithmes + GridSearchCV
5. **Évaluation** : Métriques complètes + 5-fold CV
6. **Déploiement** : Flask + Interface web

## 🎯 Technologies Utilisées

- **Python 3.8+** : Langage principal
- **scikit-learn** : Machine Learning et Pipelines
- **pandas/numpy** : Manipulation données
- **matplotlib/seaborn** : Visualisations
- **Flask** : Déploiement web
- **Tailwind CSS** : Design interface
- **Jupyter** : Développement interactif

## 📝 Prochaines Étapes

- [ ] Enregistrement vidéo démonstration (5 minutes)
- [ ] Publication sur LinkedIn
- [ ] Tests utilisateurs finaux

---

**🎓 Projet Data Science IGA** - Solution ML de bout en bout répondant à 100% des exigences du cahier des charges
