import theano
import numpy as np


class ReLUMLPLayer:
    def __init__(self, input_size, output_size):
        self.weights = 0.01 * np.random.randn(input_size, output_size)
        self.biases = np.zeros((output_size,))

    def step(self, inputs):
        ip = theano.tensor.dot(self.weights, inputs) + self.biases
        return theano.tensor.nnet.relu(ip)
