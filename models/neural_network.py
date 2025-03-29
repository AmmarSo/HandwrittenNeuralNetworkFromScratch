from models.layer import Layer
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, layers_sizes):
        self.layers = []
        for num_neurons in layers_sizes:
            layer = Layer(input_size, num_neurons)
            self.layers.append(layer)
            input_size = num_neurons

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, dL_dout, learning_rate):
        for layer in reversed(self.layers):
            dL_dout = layer.backward(dL_dout, learning_rate)

    def save_model(self, filename):
        # Sauvegarde des poids et biais dans un fichier
        model_data = []
        for layer in self.layers:
            layer_data = {
                "weights": [neuron.weights for neuron in layer.neurons],
                "biases": [neuron.bias for neuron in layer.neurons]
            }
            model_data.append(layer_data)
        np.save(filename, model_data)
        print(f"Modèle sauvegardé dans le fichier {filename}")

    def load_model(self, filename):
        # Chargement des poids et biais depuis un fichier
        model_data = np.load(filename, allow_pickle=True)
        for layer, layer_data in zip(self.layers, model_data):
            for neuron, weights, bias in zip(layer.neurons, layer_data['weights'], layer_data['biases']):
                neuron.weights = np.array(weights)
                neuron.bias = bias
        print(f"Modèle chargé depuis le fichier {filename}")
