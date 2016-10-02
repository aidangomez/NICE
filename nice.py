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
        for (i, layer) in enumerate(self.layers):
            if i % 2 == 0:
                output_1, output_2 = layer(output_1, output_2)
            else:
                output_2, output_1 = layer(output_2, output_1)

        return (output_1, output_2)


class NICELayer:
    def __init__(self,
                 coupling_layer,
                 coupling_rule):
        self.coupling_layer = coupling_layer
        self.coupling_rule = coupling_rule

    def __call__(self, input_1, input_2):
        output_1 = input_1
        output_2 = self.coupling_rule(input_2, self.coupling_layer(input_1))
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
    def __init__(self, input_size, output_size):
        self.name = uuid4()
        self.input_size = input_size
        self.output_size = output_size
        params[str(self.name)+"-W"] = theano.shared(0.01 * np.random.randn(input_size, output_size))
        params[str(self.name)+"-b"] = theano.shared(np.zeros((output_size)))

    def __call__(self, inputs):
        ip = theano.tensor.dot(inputs, params[str(self.name)+"-W"]) + params[str(self.name)+"-b"]
        return theano.tensor.nnet.relu(ip)


def log_loss(output, s):
    component_loss = -(np.log(1 + np.exp(output)) + np.log(1 + np.exp(-output)))
    return component_loss.sum() +  s.sum()

def build_model():
    x_1 = theano.tensor.matrix('x_1', dtype=theano.config.floatX)
    x_2 = theano.tensor.matrix('x_2', dtype=theano.config.floatX)

    x_1_size = 392
    x_2_size = 392
    s = np.zeros(x_1_size + x_2_size, dtype=theano.config.floatX)
    params["s"] = theano.shared(s, name="s")

    coupling_1 = CouplingLayer(x_1_size, 5, [1000]*5, x_2_size)
    nice_1 = NICELayer(coupling_1, NICELayer.additive_coupling)
    coupling_2 = CouplingLayer(x_2_size, 5, [1000]*5, x_1_size)
    nice_2 = NICELayer(coupling_2, NICELayer.additive_coupling)
    coupling_3 = CouplingLayer(x_1_size, 5, [1000]*5, x_2_size)
    nice_3 = NICELayer(coupling_3, NICELayer.additive_coupling)
    coupling_4 = CouplingLayer(x_2_size, 5, [1000]*5, x_1_size)
    nice_4 = NICELayer(coupling_4, NICELayer.additive_coupling)

    nice_stack = NICECombinationLayer([nice_1, nice_2, nice_3, nice_4])

    nice_output_1, nice_output_2 = nice_stack(x_1, x_2)
    output = theano.tensor.concatenate([nice_output_1, nice_output_2], axis=0)
    pred = output * np.exp(params["s"])

    f_pred = theano.function([x_1, x_2], pred)

    cost = log_loss(output, params["s"])

    return params, x_1, x_2, f_pred, cost
