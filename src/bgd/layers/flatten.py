# layers/flatten.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'Flatten'
]

from .layer import Layer

class Flatten(Layer):

    def __init__(self, order='C', copy=False):
        Layer.__init__(self, copy=copy, save_input=False, save_output=False)
        self.order = order
        self.in_shape = None

    def _forward(self, X):
        self.in_shape = X.shape
        return X.reshape((X.shape[0], -1), order=self.order)

    def _backward(self, error):
        return error.reshape(self.in_shape, order=self.order)

    def get_parameters(self):
        return None # Non-parametric layer
