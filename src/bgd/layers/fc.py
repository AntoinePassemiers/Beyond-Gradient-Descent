# layers/fc.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'FullyConnected'
]

import numpy as np

from bgd.initializers import ZeroInitializer, GlorotUniformInitializer
from .layer import Layer

class FullyConnected(Layer):
    """ Fully connected (dense) neural layer. Each output neuron is a
    weighted sum of its inputs with possibly a bias.

    Args:
        n_in (int):
            Number of input neurons.
        n_out (int):
            Number of output neurons.
        copy (bool):
            Whether to copy layer output.
        with_bias (bool):
            Whether add a bias to output neurons.
        dtype (type):
            Type of weights and biases.
        initializer (:class:`bgd.initializers.Initializer`):
            Initializer of the weights.
        bias_initializer (:class:`bgd.initializers.Initializer`):
            Initializer of the biases.

    Attributes:
        weights (:obj:`np.ndarray`):
            Matrix of weights.
        biases (:obj:`np.ndarray`):
            Vector of biases.
    """

    def __init__(self, n_in, n_out, copy=False, with_bias=True,
                 dtype=np.float32, initializer=GlorotUniformInitializer(),
                 bias_initializer=ZeroInitializer()):
        Layer.__init__(self, copy=copy, save_output=False)
        self.with_bias = with_bias
        self.dtype = dtype
        self.n_in = n_in
        self.n_out = n_out
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.weights = self.initializer.initialize((self.n_in, self.n_out), dtype=self.dtype)
        if self.with_bias:
            self.biases = self.bias_initializer.initialize((1, self.n_out), dtype=self.dtype)
        else:
            self.biases = None

    def _forward(self, X):
        # Output: X * W + b
        return np.dot(X, self.weights) + self.biases

    def _backward(self, error):
        gradient_weights = np.dot(self.current_input.T, error)
        if self.with_bias:
            gradient_bias = np.sum(error, axis=0, keepdims=True)
            gradients = (gradient_weights, gradient_bias)
        else:
            gradients = gradient_weights
        if self.propagate:
            signal = np.dot(error, self.weights.T)
        else:
            signal = None
        return (signal, gradients)

    def get_parameters(self):
        if self.with_bias:
            return (self.weights, self.biases)
        return (self.weights,)

    def update_parameters(self, delta_fragments):
        self.weights -= delta_fragments[0]
        if self.with_bias:
            self.biases -= delta_fragments[1]
