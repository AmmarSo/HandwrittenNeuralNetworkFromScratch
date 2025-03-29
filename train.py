import os
import numpy as np
from data.load_data import load_mnist_data
from data.preprocess import preprocess_images, one_hot_encode
from models.neural_network import NeuralNetwork

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Chargement des données MNIST")
train_images, train_labels, test_images, test_labels = load_mnist_data()

print("Prétraitement des images")
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)
train_labels = one_hot_encode(train_labels)
test_labels = one_hot_encode(test_labels)

print("Réduction des données pour un test rapide")
train_images = train_images[:60000]
train_labels = train_labels[:60000]

input_size = 784
layers_sizes = [128, 64, 10]

print("Initialisation du réseau de neurones")
network = NeuralNetwork(input_size, layers_sizes)


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss


def accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    return np.mean(predictions == labels)


def train(network, train_images, train_labels, epochs, learning_rate):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i in range(len(train_images)):
            outputs = network.forward(train_images[i])
            loss = cross_entropy_loss(train_labels[i], outputs)
            if i % 100 == 0:
                print(f"  Image {i}/{len(train_images)}, Loss: {loss:.4f}")


def evaluate(network, test_images, test_labels):
    correct_predictions = 0
    total_images = len(test_images)

    for i in range(total_images):
        outputs = network.forward(test_images[i])
        prediction = np.argmax(outputs)
        true_label = np.argmax(test_labels[i])
        if prediction == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_images
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")


print("Début de l'entraînement")
train(network, train_images, train_labels, epochs=100, learning_rate=0.01)

print("Évaluation du réseau")
evaluate(network, test_images, test_labels)

print("Fin du script")
