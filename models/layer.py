from models.neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, input_size, num_neurons):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]
        self.last_inputs = None

    def forward(self, inputs):
        self.last_inputs = inputs
        outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return np.array(outputs)

    def backward(self, dL_dout, learning_rate):
        dL_dinputs = np.zeros_like(self.last_inputs)
        for neuron, grad in zip(self.neurons, dL_dout):
            neuron_gradient = neuron.backward(grad, learning_rate)
            if np.any(np.isnan(neuron_gradient)) or np.any(np.isinf(neuron_gradient)):
                raise ValueError("NaN or Inf encountered in gradients!")
            dL_dinputs += neuron_gradient
        return dL_dinputs
