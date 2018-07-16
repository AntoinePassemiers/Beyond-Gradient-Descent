""" This module contains all the layers that are implemented.
Any layer needs to inherit from :class:`bgd.layers.Layer` and
to implement its abstract methods (:obj:`_forward`, :obj:`backward`
and :obj:`get_parameters`, even by returning None if layer is
non-parametric). """

# layers.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'Activation', 'Dropout', 'GaussianNoise', 'Flatten', 'Lambda'
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
            grad_X[:] = (grad_X >= 0)
        elif self.function == Activation.SOFTMAX:
            grad_X = X * (1. - X)
        else:
            raise NotImplementedError()
        return grad_X * error

    def get_parameters(self):
        return None # Non-parametric layer


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


class Lambda(Layer):

    def __init__(self, forward_op, backward_op, copy=False):
        Layer.__init__(self, copy=copy)
        self.forward_op = forward_op
        self.backward_op = backward_op

    def _forward(self, X):
        return self.forward_op(X)

    def _backward(self, error):
        return self.backward_op(error)

    def get_parameters(self):
        return None # Non-parametric layer

