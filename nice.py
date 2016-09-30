from collections import OrderedDict
from uuid import uuid4

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import data


params = OrderedDict()

class NICECombinationLayer:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, input_1, input_2):
        output_1, output_2 = (input_1, input_2)
        for (i, layer) in enumerate(layers):
            if i % 2 == 0:
                output_1, output_2 = layer(output_1, output_2)
            else:
                output_2, output_1 = layer(output_2, output_1)

        return (output_1, output_2)


class NICELayer:
    def __init__(self,
                 coupling_layer,
                 coupling_rule=additive_coupling):
        self.couple = coupling_layer
        self.coupling_f = coupling_function

    def __call__(self, input_1, input_2):
        output_1 = input_1
        output_2 = self.couple(input_2, self.coupling_f(input_1))
        return (output_1, output_2)

    def additive_coupling(a, b):
        return a + b

    def multiplicative_coupling(a, b):
        return a * b


class CouplingLayer:
    def __init__(self, input_size, depth, widths, output_size):
        self.layers = []
        layer_input_size = input_size
        for d in range(depth):
            layer = ReLUMLPLayer(layer_input_size, widths[d])
            layer_input_size = widths[d]
            self.layers.append(layer)

        layer = ReLUMLPLayer(layer_input_size, output_size)
        self.layers.append(layer)

    def __call__(self, input):
        layer_output = input
        for layer in self.layers:
            layer_output = layer(layer_output)

        return layer_output


class ReLUMLPLayer:
    def __init__(self, input_size, output_size, name=uuid4()):
        self.name = name
        params[str(self.name)+"-W"] = theano.tensor.shared(0.01 * np.random.randn(input_size, output_size))
        params[str(self.name)+"-b"] = theano.tensor.shared(np.zeros((output_size,)))

    def __call__(self, inputs):
        ip = theano.tensor.dot(params[str(self.name)+"-W"], inputs) + params[str(self.name)+"-b"]
        return theano.tensor.nnet.relu(ip)


def log_loss(output, s):
    component_loss = -(np.log(1 + exp(output)) + np.log(1 + exp(-output)))
    return component_loss.sum() +  np.sum(s))

def build_model():
    x_1 = tensor.vector('x', dtype='int8')
    x_2 = tensor.vector('x', dtype='int8')
    y_1 = tensor.vector('y', dtype='int8')
    y_2 = tensor.vector('y', dtype='int8')

    params["s"] = theano.tensor.shared(np.zeros((y_1.shape[0] + y_2.shape[0])))

    coupling_1 = CouplingLayer(x_1.shape[0], 5, 1000, x_2.shape[0])
    nice_1 = NICELayer(coupling_1)
    coupling_2 = CouplingLayer(x_2.shape[0], 5, 1000, x_1.shape[0])
    nice_2 = NICELayer(coupling_2)
    coupling_3 = CouplingLayer(x_1.shape[0], 5, 1000, x_2.shape[0])
    nice_3 = NICELayer(coupling_3)
    coupling_4 = CouplingLayer(x_2.shape[0], 5, 1000, x_1.shape[0])
    nice_4 = NICELayer(coupling_4)

    nice_stack = NICECombinationLayer([nice_1, nice_2, nice_3, nice_4])

    nice_output_1, nice_output_2 = nice_stack(x_1, x_2)
    pred = np.concatenate([nice_output_1, nice_output_2], axis=1) * np.exp(params["s"])

    f_pred = theano.function([x_1, x_2], pred)

    cost = log_loss(output, params["s"])

    return x_1, x_2, y_1, y_2, f_pred_prob, f_pred, cost
