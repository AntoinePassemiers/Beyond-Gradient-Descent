# layers/droupout.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'Dropout'
]

import numpy as np

from .layer import Layer

class Dropout(Layer):

    def __init__(self, keep_proba=.5, copy=False):
        Layer.__init__(self, copy=copy, save_input=False, save_output=False)
        self.keep_proba = keep_proba
        self.active = False
        self.mask = None

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def _forward(self, X):
        if self.active:
            self.mask = (np.random.rand(*X.shape) > (1. - self.keep_proba))
            return self.mask * X
        return X

    def _backward(self, error):
        assert self.active
        return self.mask * error

    def get_parameters(self):
        return None # Non-parametric layer
