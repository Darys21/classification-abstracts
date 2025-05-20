# Projet de Classification d'Abstracts Académiques

## Auteur
**ANGUILET Joan-Yves Darys**  
Étudiant en Master 2 - Promotion Janvier 2025

## Table des matières
- [Vue d'ensemble](#vue-densemble)
- [Jeu de données](#jeu-de-données)
- [Installation](#installation)
- [Entraînement du modèle](#entraînement-du-modèle)
- [Interface de démonstration](#interface-de-démonstration)
- [Structure du projet](#structure-du-projet)
- [Dépendances](#dépendances)
- [Améliorations futures](#améliorations-futures)

## Vue d'ensemble

Ce projet consiste en la classification automatique d'abstracts académiques dans six catégories distinctes à l'aide d'un modèle BERT (Bidirectional Encoder Representations from Transformers). Le projet comprend :

- Un notebook Jupyter (`classification_project.ipynb`) pour l'exploration des données et l'entraînement du modèle
- Un script de démonstration (`demo.py`) avec une interface utilisateur interactive
- Un modèle pré-entraîné sauvegardé
- Ce fichier README documentant le projet

## Jeu de données

Le jeu de données utilisé pour ce projet contient des abstracts académiques étiquetés selon six catégories :

1. Computer Science
2. Physics
3. Mathematics
4. Statistics
5. Quantitative Biology
6. Quantitative Finance

### Pré-traitement

Les données subissent les transformations suivantes :
- Nettoyage des valeurs manquantes
- Tokenisation avec le tokenizer BERT (bert-base-uncased)
- Découpage ou troncature à 512 tokens (longueur maximale supportée par BERT)
- Conversion des étiquettes textuelles en indices numériques (0-5)

## Installation

### Prérequis

- Python 3.7 ou supérieur
- pip (gestionnaire de paquets Python)

### Configuration de l'environnement

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/Darys21/classification-abstracts.git
   cd Projet_classification
   ```

2. **Créer et activer un environnement virtuel (recommandé)** :
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Sur Windows
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

## Entraînement du modèle

Le modèle peut être entraîné en exécutant le notebook `classification_project.ipynb` ou le script `classification_project.py`. Le processus d'entraînement comprend :

- Chargement et prétraitement des données
- Division des données en ensembles d'entraînement et de validation
- Initialisation du modèle BERT pour la classification
- Entraînement avec validation croisée
- Sauvegarde du modèle entraîné

Pour exécuter l'entraînement :
```bash
python classification_project.py
```

## Interface de démonstration

Le projet inclut une interface utilisateur interactive construite avec Gradio qui permet de :

1. Saisir un abstract académique
2. Visualiser les probabilités pour chaque catégorie
3. Obtenir la catégorie prédite

Pour lancer l'interface :

```bash
python demo.py
```

L'interface sera accessible à l'adresse : `http://127.0.0.1:7860/`

## Structure du projet

```
Projet_classification/
├── model/
│   └── abstract_classifier.pt    # Modèle entraîné
├── classification_project.ipynb   # Notebook d'entraînement
├── classification_project.py      # Script d'entraînement
├── demo.py                       # Interface de démonstration
├── README.md                     # Ce fichier
└── requirements.txt              # Dépendances du projet
```

## Dépendances

Les principales dépendances du projet sont :

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Pandas
- scikit-learn
- Gradio

