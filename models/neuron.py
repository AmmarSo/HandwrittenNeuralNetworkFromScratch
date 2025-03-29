import numpy as np


class Neuron:
    def __init__(self, input_size):
        # Initialisation de He pour les poids
        self.weights = np.random.randn(input_size) * np.sqrt(2. / input_size)
        self.bias = 0.0  # Initialiser les biais à zéro
        self.last_inputs = None
        self.last_z = None

    def activation_function(self, x):
        # Fonction d'activation ReLU
        return np.maximum(0, x)

    def forward(self, inputs):
        self.last_inputs = np.array(inputs)
        self.last_z = np.dot(self.weights, self.last_inputs) + self.bias
        return self.activation_function(self.last_z)

    def backward(self, dL_dout, learning_rate):
        # Calcul de la dérivée de la fonction d'activation (ReLU)
        dL_dz = dL_dout * (1 if self.last_z > 0 else 0)

        # Gradient par rapport aux poids
        dL_dw = dL_dz * self.last_inputs
        dL_db = dL_dz  # Gradient par rapport au biais

        # Mise à jour des poids et du biais
        self.weights -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db

        # Calcul du gradient pour les entrées (pour la couche précédente)
        dL_dx = self.weights * dL_dz
        return dL_dx
