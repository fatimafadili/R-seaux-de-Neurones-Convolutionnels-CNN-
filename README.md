Convolutional Neural Networks (CNNs) - Projet Éducatif
https://img.shields.io/badge/TensorFlow-2.x-orange
https://img.shields.io/badge/Python-3.8+-blue
https://img.shields.io/badge/Jupyter-Notebook-red
https://img.shields.io/badge/License-MIT-green

📋 Table des Matières
Introduction

Structure du Projet

Installation

Contenu Détaillé

Applications Pratiques

Résultats

Utilisation

Contribuer

Licence

🎯 Introduction
Ce projet éducatif présente une introduction complète aux Réseaux de Neurones Convolutionnels (CNN) à travers des explications théoriques et des implémentations pratiques. Le notebook guide les étudiants depuis les concepts fondamentaux jusqu'aux applications avancées de la vision par ordinateur.

Objectifs Pédagogiques
✅ Comprendre les principes de base des convolutions

✅ Maîtriser les architectures CNN standards

✅ Appliquer les CNN à des tâches réelles

✅ Explorer la segmentation et détection d'objets

✅ Comparer différentes architectures de réseaux

🏗️ Structure du Projet
text
CNN_24_25_Version_Etu.ipynb/
│
├── 1️⃣ INTRODUCTION AUX CNN
│   ├── 1.1 Pourquoi les CNN ?
│   ├── 1.2 Principe de base des convolutions
│   └── 1.3 Exemple simple avec TensorFlow
│
├── 2️⃣ APPLICATIONS PRATIQUES
│   ├── 2.1 Convolution avec plusieurs filtres
│   ├── 2.2 Fonctions d'activation (ReLU)
│   ├── 2.3 Couches de Pooling
│   └── 2.4 Architecture complète
│
├── 3️⃣ CLASSIFICATION D'IMAGES
│   ├── 3.1 MNIST - Chiffres manuscrits
│   ├── 3.2 CIFAR-10 - Objets divers
│   └── 3.3 Comparaison ANN vs CNN
│
├── 4️⃣ SEGMENTATION D'IMAGES
│   ├── 4.1 Méthodes traditionnelles
│   │   ├── Seuillage (Thresholding)
│   │   ├── K-means clustering
│   │   └── Détection de contours
│   ├── 4.2 Segmentation avec CNN
│   └── 4.3 U-Net sur Oxford-IIIT Pets
│
├── 5️⃣ DÉTECTION D'OBJETS
│   ├── 5.1 SSD (Single Shot Detector)
│   ├── 5.2 PASCAL VOC 2007
│   ├── 5.3 Métriques d'évaluation (IoU)
│   └── 5.4 Modèles avancés (YOLO/Detectron2)
│
└── 6️⃣ EXERCICES ET PROJETS
    ├── Exercice 1h : Consolidation des concepts
    └── Mini-projet 1 semaine : Détection avancée
⚙️ Installation
Prérequis
Python 3.8 ou supérieur

Jupyter Notebook ou Google Colab

8 Go de RAM minimum (recommandé)

GPU (optionnel mais recommandé pour l'entraînement)

Installation des Dépendances
bash
# Cloner le repository
git clone https://github.com/votre-username/cnn-educational-project.git
cd cnn-educational-project

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
requirements.txt :

text
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
opencv-python>=4.5.0
pillow>=9.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
tensorflow-datasets>=4.5.0
📚 Contenu Détaillé
Partie 1 : Fondamentaux des CNN
1.1 Convolution Simple
python
# Exemple de convolution avec TensorFlow
import tensorflow as tf
import numpy as np

# Image 5x5 et filtre 3x3
image = np.array([[1,2,3,4,5], ...])
kernel = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])

# Application de la convolution
convolved = tf.nn.conv2d(image_tf, kernel_tf, strides=[1,1,1,1], padding='VALID')
1.2 Filtres Multiples
Filtre Horizontal : Détection des bords horizontaux

Filtre Vertical : Détection des bords verticaux

Filtre Sobel : Détection améliorée des contours

Filtre Laplacien : Détection des changements brusques

Filtre Flou : Lissage d'image

Filtre Sharpening : Renforcement des détails

Partie 2 : Architectures CNN
2.1 Bloc CNN Standard
python
model = tf.keras.Sequential([
    # Couche Convolutionnelle
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    
    # Pooling
    tf.keras.layers.MaxPooling2D((2,2)),
    
    # Couches supplémentaires
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    # Classification
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
2.2 Améliorations Avancées
Batch Normalization : Stabilisation de l'entraînement

Dropout : Réduction du surapprentissage

Data Augmentation : Augmentation artificielle des données

Learning Rate Scheduling : Ajustement dynamique du taux d'apprentissage

🚀 Applications Pratiques
Application 1 : Classification MNIST
Objectif : Reconnaître les chiffres manuscrits
Architecture : CNN simple
Performance : ~99% de précision
Temps d'entraînement : 5 minutes sur CPU

Application 2 : Classification CIFAR-10
Objectif : Classifier 10 catégories d'objets
Architecture : CNN amélioré avec Dropout
Performance : ~75% de précision
Temps d'entraînement : 30 minutes sur GPU

Application 3 : Segmentation Oxford-IIIT Pets
Objectif : Segmenter les chats et chiens
Architecture : U-Net
Performance : IoU > 0.7
Temps d'entraînement : 2 heures sur GPU

Application 4 : Détection PASCAL VOC
Objectif : Détecter 20 classes d'objets
Architecture : SSD avec MobileNetV2
Performance : IoU variable selon la classe
Temps d'entraînement : 1 heure sur GPU

📊 Résultats
Comparaison des Performances
Modèle	Dataset	Précision	IoU	Temps d'entraînement
ANN Simple	MNIST	97%	-	2 min
CNN Simple	MNIST	99%	-	5 min
CNN Amélioré	CIFAR-10	75%	-	30 min
U-Net	Oxford Pets	-	0.72	2 h
SSD	PASCAL VOC	-	0.45-0.70	1 h
Visualisations
Filtres Appris : Visualisation des patterns appris

Feature Maps : Activation des différentes couches

Courbes d'Apprentissage : Suivi de la perte et précision

Prédictions : Comparaison avec les véritables labels

🎮 Utilisation
Exécution Complète
bash
# Ouvrir le notebook
jupyter notebook CNN_24_25_Version_Etu.ipynb

# Ou utiliser Google Colab
# Télécharger le notebook et l'ouvrir dans Colab
Exécution Section par Section
Section 1 : Concepts fondamentaux (30 min)

Section 2 : Applications de base (45 min)

Section 3 : Classification (1 h)

Section 4 : Segmentation (1.5 h)

Section 5 : Détection d'objets (2 h)

Section 6 : Projets pratiques (variable)

Pour les Enseignants
python
# Configuration recommandée pour la classe
config = {
    "sections_par_cours": 2,
    "durée_totale": "6 séances de 3h",
    "prérequis": "Python, Algèbre linéaire",
    "matériel": "Colab Pro recommandé",
    "évaluation": "Projet final + exercices"
}
🤝 Contribuer
Les contributions sont les bienvenues ! Voici comment contribuer :

Fork le projet

Clone votre fork

Créez une branche pour votre fonctionnalité

Commitez vos changements

Push vers votre branche

Créez une Pull Request

Guide de Style
Code commenté en français ou anglais

Documentation claire et complète

Tests pour les nouvelles fonctionnalités

Respect des conventions PEP 8

📄 Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

text
MIT License

Copyright (c) 2024 [fatima el fadili]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
👥 Auteurs
Votre Nom - Développement initial - votre-email@domaine.com

Contributeurs - Voir la liste des contributeurs

🙏 Remerciements
TensorFlow Team pour l'excellente documentation

Google Colab pour les ressources de calcul

Communauté Open Source pour les datasets et outils

Étudiants pour les retours et améliorations

📚 Références
Deep Learning - Ian Goodfellow

TensorFlow Documentation

CS231n - Stanford University

Papers with Code
