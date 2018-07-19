# layers/activation.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'Activation'
]

import numpy as np

from .layer import Layer

class Activation(Layer):

    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    RELU = 'relu'
    SOFTMAX = 'softmax'

    def __init__(self, function='sigmoid', copy=False):
        Layer.__init__(self, copy=copy)
        self.function = function.lower()
        self.copy = copy

    def _forward(self, X):
        if self.function == Activation.SIGMOID:
            out = 1. / (1. + np.exp(-X))
        elif self.function == Activation.TANH:
            out = np.tanh(X)
        elif self.function == Activation.RELU:
            out = np.maximum(X, 0)
        elif self.function == Activation.SOFTMAX:
            e = np.exp(X)
            out = e / np.sum(e, axis=1, keepdims=True)
        else:
            raise NotImplementedError()
        return out

    def _backward(self, error):
        X = self.current_output
        if self.function == Activation.SIGMOID:
            grad_X = X * (1. - X)
        elif self.function == Activation.TANH:
            grad_X = 1. - X ** 2
        elif self.function == Activation.RELU:
            grad_X = self.current_input
            if self.copy:
                grad_X = np.empty_like(grad_X)
            grad_X[:] = (grad_X > 0)
        elif self.function == Activation.SOFTMAX:
            grad_X = X * (1. - X)
        else:
            raise NotImplementedError()
        return grad_X * error

    def get_parameters(self):
        return None # Non-parametric layer
