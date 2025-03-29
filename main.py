import os
import numpy as np
from data.load_data import load_mnist_data
from data.preprocess import preprocess_images, one_hot_encode
from models.neural_network import NeuralNetwork

# Désactivation des optimisations oneDNN si nécessaire
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Chargement des données MNIST
print("Chargement des données MNIST")
train_images, train_labels, test_images, test_labels = load_mnist_data()

# Prétraitement des images
print("Prétraitement des images")
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

# Initialisation des paramètres du réseau
input_size = 784  # 28*28 pixels
layers_sizes = [350, 10]  # Une couche cachée et une couche de sortie

# Initialisation du réseau de neurones
print("Initialisation du réseau de neurones")
network = NeuralNetwork(input_size, layers_sizes)


# Définir la fonction de perte (entropie croisée)
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss


# Définir la fonction de précision
def accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)


# Implémenter la fonction d'entraînement
def train(network, train_images, train_labels, epochs, learning_rate):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Boucle sur chaque image et label de l'ensemble d'entraînement
        for i in range(len(train_images)):
            # Étape 1: Propagation Avant (Forward Pass)
            outputs = network.forward(train_images[i])

            # Étape 2: Calcul de la Perte
            loss = cross_entropy_loss(train_labels[i], outputs)

            # Afficher la perte périodiquement pour suivre l'entraînement
            if i % 100 == 0:
                print(f"  Image {i}/{len(train_images)}, Loss: {loss:.4f}")

            # Étape 3: Calcul des Gradients de la Perte par rapport à la Sortie
            dL_dout = outputs - train_labels[i]

            # Étape 4: Rétropropagation (Backward Pass)
            network.backward(dL_dout, learning_rate)

    print("Entraînement terminé.")


# Fonction pour sauvegarder le modèle
def save_model(network, filename):
    model_data = []
    for layer in network.layers:
        layer_data = {
            "weights": [neuron.weights for neuron in layer.neurons],
            "biases": [neuron.bias for neuron in layer.neurons]
        }
        model_data.append(layer_data)
    np.save(filename, model_data)
    print(f"Modèle sauvegardé dans le fichier {filename}")

# Fonction pour charger le modèle (optionnel, pour référence future)
def load_model(network, filename):
    model_data = np.load(filename, allow_pickle=True)
    for layer, layer_data in zip(network.layers, model_data):
        for neuron, weights, bias in zip(layer.neurons, layer_data['weights'], layer_data['biases']):
            neuron.weights = np.array(weights)
            neuron.bias = bias
    print(f"Modèle chargé depuis le fichier {filename}")


# Entraînement du réseau
epochs = 3  # Définir le nombre d'epochs
learning_rate = 0.001  # Définir le taux d'apprentissage
print("Début de l'entraînement")
train(network, train_images, train_labels, epochs, learning_rate)

# Sauvegarde du modèle après l'entraînement
save_model(network, "mnist_model.npy")

# Évaluation du réseau
print("Évaluation du réseau sur le jeu de test")

# Propagation avant sur chaque image de test
test_outputs = []
for image in test_images:
    output = network.forward(image)
    test_outputs.append(output)

# Convertir la liste en tableau NumPy
test_outputs = np.array(test_outputs)

# Calcul de la précision
test_accuracy = accuracy(test_labels, test_outputs)
print(f"Précision sur le jeu de test: {test_accuracy * 100:.2f}%")
