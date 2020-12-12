import numpy as np

X = np.random.uniform(-1, 1, (4, 4))

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 4)

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)