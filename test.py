import os
import numpy as np
from PIL import Image
from data.load_data import load_mnist_data
from data.preprocess import preprocess_images, one_hot_encode
from models.neural_network import NeuralNetwork

# Désactivation des optimisations oneDNN si nécessaire
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Définir les paramètres du réseau
input_size = 784  # 28*28 pixels
layers_sizes = [350, 10]  # Une couche cachée et une couche de sortie

# Initialisation du réseau de neurones
print("Initialisation du réseau de neurones")
network = NeuralNetwork(input_size, layers_sizes)

# Charger le modèle pré-entraîné
print("Chargement du modèle")
network.load_model("mnist_model.npy")

# Charger et prétraiter l'image du chiffre manuscrit
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
    image = image.resize((28, 28))  # Redimensionner à 28x28 pixels
    image_array = np.array(image) / 255.0  # Normaliser les pixels
    image_array = image_array.flatten()  # Aplatir l'image en un vecteur
    return image_array

# Charger l'image
image_path = "digit.png"  # Remplace par le chemin vers l'image téléchargée
image_array = preprocess_image(image_path)

# Faire une prédiction
prediction = network.forward(image_array)

# Trouver la classe prédite (le chiffre avec la plus haute probabilité)
predicted_digit = np.argmax(prediction)
print(f"Le modèle prédit que le chiffre est : {predicted_digit}")

