# layers/max_pooling2d.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'MaxPooling2D'
]

import numpy as np

from .layer import Layer
# pylint: disable=import-error,no-name-in-module
from .max_pooling import max_pooling_2d_backward, max_pooling_2d_forward

class MaxPooling2D(Layer):

    def __init__(self, pool_shape, strides=(1, 1), copy=False):
        Layer.__init__(self, copy=copy, save_input=False, save_output=False)
        self.pool_shape = pool_shape
        self.strides = strides
        self.mask = None
        self.out_buffer = None
        self.in_buffer = None

    def _forward(self, X):
        if self.out_buffer is None or X.shape[0] > self.out_buffer.shape[0]:
            out_height = (X.shape[1] - self.pool_shape[0] + 1) // self.strides[0]
            out_width = (X.shape[2] - self.pool_shape[1] + 1) // self.strides[1]
            self.out_buffer = np.empty((X.shape[0], out_height, out_width, X.shape[3]),
                                       dtype=X.dtype)
            self.in_buffer = np.empty(X.shape, dtype=X.dtype)
            self.mask = np.empty(X.shape, dtype=np.int8)
        max_pooling_2d_forward(self.out_buffer, self.mask, X, self.pool_shape, self.strides)
        return self.out_buffer[:X.shape[0], :, :, :]

    def _backward(self, error):
        max_pooling_2d_backward(self.in_buffer, error, self.mask, self.pool_shape, self.strides)
        return self.in_buffer[:error.shape[0], :, :, :]

    def get_parameters(self):
        return None # Non-parametric layer
