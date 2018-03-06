# -*- coding: utf-8 -*-
# layers.py
# author : Antoine Passemiers, Robin Petit

from bgd.initializers import *
from bgd.operators import *

from abc import ABCMeta, abstractmethod
import numpy as np


class Layer(metaclass=ABCMeta):

    def __init__(self, copy=True):
        self.copy = copy
        self.current_input = None
        self.current_output = None
    
    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def _forward(self, X):
        pass
    
    @abstractmethod
    def _backward(self, X):
        pass

    def forward(self, X):
        self.current_input = X
        self.current_output = self._forward(X)
        if self.copy:
            if np.may_share_memory(X, self.current_output, np.core.multiarray.MAY_SHARE_BOUNDS):
                self.current_output = np.copy(self.current_output)
        return self.current_output
    
    def backward(self, *args, **kwargs):
        return self._backward(*args, **kwargs)


class FullyConnected(Layer):

    def __init__(self, n_in, n_out, copy=True, with_bias=True, dtype=np.float32, initializer=GaussianInitializer(0, .01)):
        Layer.__init__(self, copy=copy)
        self.with_bias = with_bias
        self.dtype = dtype
        self.n_in = n_in
        self.n_out = n_out
        self.initializer = initializer
        self.weights = self.initializer.initialize((self.n_in, self.n_out), dtype=self.dtype)
        if self.with_bias:
            self.biases = np.zeros((1, self.n_out), dtype=self.dtype)
        else:
            self.biases = None

    def get_parameters(self):
        return (self.weights, self.biases) if self.with_bias else (self.weights,)

    def _forward(self, X):
        return np.dot(X, self.weights) + self.biases
    
    def _backward(self, error, extra_info={}):
        gradient_weights = np.dot(self.current_input.T, error)
        gradient_weights += extra_info['l2_reg'] * self.weights # Derivative of L2 regularization term
        gradient_bias = np.sum(error, axis=0, keepdims=True)
        self.update(gradient_weights, gradient_bias, extra_info['learning_rate'])
        error = np.dot(error, self.weights.T)
        return error

    def update(self, dW, db, learning_rate):
        self.weights -= learning_rate * dW
        if self.with_bias:
            self.biases -= learning_rate * np.sum(db, axis=0, keepdims=True)


class Activation(Layer):

    def __init__(self, function='sigmoid', copy=True):
        Layer.__init__(self, copy=copy)
        self.function = function.lower()
        self.copy = copy

    def get_parameters(self):
        return None

    def _forward(self, X):
        if self.function == 'sigmoid':
            return 1. / (1. + np.exp(-X))
        elif self.function == 'relu':
            return np.maximum(X, 0)
        elif self.function == 'softmax':
            e = np.exp(X)
            return e / np.sum(e, axis=1, keepdims=True)
        else:
            raise NotImplementedError()
    
    def _backward(self, X, extra_info={}):
        if self.function == 'sigmoid':
            return X * (1. - X)
        elif self.function == 'relu':
            if self.copy:
                X = np.copy(X)
            X[self.current_output <= 0] = 0
            return X
        elif self.function == 'softmax':
            return X
        else:
            raise NotImplementedError()


class Convolutional2D(Layer):

    def __init__(self, filter_shape, n_filters, strides=[1, 1], with_bias=True, copy=True, initializer=GaussianInitializer(0, .01)):
        Layer.__init__(self, copy=copy)
        self.filter_shape = filter_shape
        self.strides = strides
        self.initializer = initializer
        self.with_bias = with_bias
        self.n_filters = n_filters
        assert(len(filter_shape) == 3)
        self.filters = None
        self.biases = None

    def init_weights(self, dtype):
        filters_shape = tuple([self.n_filters] + list(self.filter_shape))
        self.filters = self.initializer.initialize(filters_shape, dtype=dtype)
        self.biases = np.zeros(self.n_filters, dtype=dtype)

    def get_parameters(self):
        return (self.filters, self.biases) if self.with_bias else (self.filters,)

    def _forward(self, X):
        if X.ndim == 3:
            X = X[..., np.newaxis]
        if self.filters is None:
            self.init_weights(X.dtype)
        # TODO: apply convolution with numpy or cython ?
        output = conv_2d_forward(X, self.filters, self.biases, self.strides, self.with_bias)
        return output
    
    def _backward(self, error, extra_info={}):
        # TODO: Deep philosophical thoughts on backward convolution
        gradient_bias = np.sum(error, axis=(0, 2, 3))
        # TODO: compute gradient_filters and call update
        return error

    def update(self, dW, db, learning_rate):
        self.filters -= learning_rate * dW
        if self.with_bias:
            self.biases -= learning_rate * np.sum(db, axis=0, keepdims=True)


class Dropout(Layer):

    def __init__(self, keep_proba=.5, copy=True):
        Layer.__init__(self, copy=copy)
        self.keep_proba = keep_proba
        self.active = False
        self.mask = None
    
    def activate(self):
        self.active = True
    
    def deactivate(self):
        self.active = False
    
    def get_parameters(self):
        return None
    
    def _forward(self, X):
        if self.active:
            self.mask = (np.random.rand(*X.shape) > (1. - self.keep_proba))
            return self.mask * X
        else:
            return X
    
    def _backward(self, X, extra_info={}):
        assert(self.active)
        return self.mask * X


class Flatten(Layer):

    def __init__(self, order='C', copy=True):
        Layer.__init__(self, copy=copy)
        self.order = order
        self.in_shape = None
    
    def get_parameters(self):
        return None
    
    def _forward(self, X):
        self.in_shape = X.shape
        return X.reshape((X.shape[0], -1), order=self.order)
    
    def _backward(self, X, extra_info={}):
        return X.reshape(self.in_shape, order=self.order)
