# layers/noise.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'GaussianNoise'
]

import numpy as np

from .layer import Layer

class GaussianNoise(Layer):

    def __init__(self, mean, stdv, clip=None, copy=False):
        Layer.__init__(self, copy=copy, save_input=False, save_output=False)
        self.save_input = False
        self.save_output = False
        self.mean = mean
        self.stdv = stdv
        self.clip = clip

    def _forward(self, X):
        noised_X = X + np.random.normal(self.mean, self.stdv, size=X.shape)
        if self.clip:
            noised_X = np.clip(noised_X, self.clip[0], self.clip[1])
        return noised_X

    def _backward(self, error):
        return error

    def get_parameters(self):
        return None # Non-parametric layer
