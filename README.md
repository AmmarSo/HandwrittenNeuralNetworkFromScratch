# Neural Network from Scratch - MNIST Digit Recognition

## This project implements a neural network from scratch (without using machine learning libraries like TensorFlow or PyTorch) to recognize handwritten digits from the MNIST dataset.

Read the full article on Medium

The dataset is loaded using:
from tensorflow.keras.datasets import mnist

## Project Structure

├── src/                # Code source du réseau de neurones
│   ├── neuron.py       # Classe Neuron
│   ├── layer.py        # Classe Layer
│   ├── network.py      # Architecture du réseau et entraînement
│   ├── train.py        # Boucle d'entraînement
│   └── test.py         # Code d'évaluation
├── data/               # (Optionnel) Chargement ou stockage des données
├── notebooks/          # Notebooks Jupyter (pour les expérimentations ou visualisations)
├── models/             # Poids du modèle sauvegardés (optionnel)
├── README.md           # Ce fichier
├── requirements.txt    # Dépendances Python
└── .gitignore          # Fichiers à ignorer par Git



## Example Results

| Epoch | Learning Rate | Layers               | Accuracy |
|-------|---------------|----------------------|----------|
| 1     | 0.001         | [256, 128, 64]       | 82.06%   |
| 1     | 0.001         | [350]                | 93.30%   |
| 1     | 0.001         | [256, 128, 64, 32]   | 75.48%   |
| 2     | 0.001         | [350]                | 94.66%   |
| 10    | 0.001         | [350]                | 87.77%   |
| 1     | 0.0001        | [350]                | 76.05%   |
| 3     | 0.001         | [350]                | 95.55%   |
| 4     | 0.001         | [350]                | 77.20%   |


## Technologies Used

Python 3.11

NumPy

MNIST Dataset (via tensorflow.keras.datasets)

## Core Concepts

- Forward Propagation

- Backpropagation

- Gradient Descent

- Weight & Bias Initialization

- Cost Function (Mean Squared Error)

## License

MIT License © 2025 Ammar Souchon

## Author

AmmarSo Follow me on Medium 

## Show your support

If you like this project, please consider leaving a ⭐ on the repository!
