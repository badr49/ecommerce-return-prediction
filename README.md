### Objectif du Projet

**Problématique métier définie** : 
- **Contexte** : Le commerce électronique connaît une croissance exponentielle, mais les retours de produits représentent un défi majeur pour les entreprises. Les retours génèrent des coûts logistiques, affectent la satisfaction client et impactent la rentabilité.
- **Variable cible** : `will_return` (binaire) : 1 si le produit sera retourné, 0 sinon
- **Intérêt du Machine Learning** : Réduction des coûts logistiques, optimisation du stockage, amélioration de l'expérience client, prise de décision proactive


### Pipeline Complet
1. **Chargement** : Dataset
2. **Nettoyage** : Gestion valeurs manquantes/aberrantes
3. **Prétraitement** : Encodage + Standardisation
4. **Modélisation** : 3 algorithmes + GridSearchCV
5. **Évaluation** : Métriques complètes + 5-fold CV
6. **Déploiement** : Flask + Interface web

## Technologies Utilisées

- **Python 3.8+** : Langage principal
- **scikit-learn** : Machine Learning et Pipelines
- **pandas/numpy** : Manipulation données
- **matplotlib/seaborn** : Visualisations
- **Flask** : Déploiement web
- **Tailwind CSS** : Design interface
- **Jupyter** : Développement interactif
