# -*- coding: utf-8 -*-
# layers.py
# author : Antoine Passemiers, Robin Petit

from bgd.initializers import *

from abc import ABCMeta, abstractmethod
import numpy as np


class Layer(metaclass=ABCMeta):

    def __init__(self):
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
        self.current_input = X # TODO: Copy X ?
        self.current_output = self._forward(X)
        assert(not np.shares_memory(X, self.current_output, np.core.multiarray.MAY_SHARE_BOUNDS))
        return self.current_output
    
    def backward(self, *args, **kwargs):
        return self._backward(*args, **kwargs)


class FullyConnected(Layer):

    def __init__(self, n_in, n_out, with_bias=True, dtype=np.float32, initializer=GaussianInitializer(0, .01)):
        Layer.__init__(self)
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
        Layer.__init__(self)
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
            if self.copy:
                X = np.copy(X)
            return X
        else:
            raise NotImplementedError()
