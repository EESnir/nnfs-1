import numpy as np 
import nnfs
from nnfs.datasets import spiral_data  # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c

nnfs.init()

X, y = spiral_data(100, 3)   


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


def run_layer_chain(intput_vec, layerobj_list, activation_obj):
    temp_output = intput_vec
    for layer in layerobj_list:
        layer.forward(temp_output)
        temp_output = layer.output
        activation_obj.forward(temp_output)
        temp_output = activation_obj.output
    return temp_output


layer1 = Layer_Dense(2,5)
layer2 = Layer_Dense(5,5)

l_list = [layer1, layer2]
# activation1 =
print(run_layer_chain(X, l_list, Activation_ReLU()))
# layer1.forward(X)

#print(layer1.output)
# activation1.forward(layer1.output)
# print(activation1.output)


