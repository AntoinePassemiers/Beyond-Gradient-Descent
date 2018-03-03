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
    
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    def forward(self, X):
        self.current_input = X # TODO: Copy X ?
        self.current_output = self._forward(X)
        assert(not np.shares_memory(X, self.current_output, np.core.multiarray.MAY_SHARE_BOUNDS))
        return self.current_output


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
    
    def _backward(self, X):
        pass

    def update(self, dW, delta, learning_rate, reg_L2):
        self.weights -= learning_rate * (dW + reg_L2 * self.weights)
        if self.with_bias:
            self.biases -= np.sum(delta, axis=0, keepdims=True)


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
    
    def _backward(self, X):
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

    def update(self, *args, **kwargs):
        pass


if __name__ == '__main__':

    initializer = GaussianInitializer(0, .001)
    l1 = FullyConnected(64, 500, initializer=initializer)
    l2 = Activation(l1, function='sigmoid')
    l3 = FullyConnected(500, 10, l2, initializer=initializer)
    l4 = Activation(l3, function='sigmoid')

    for i in range(1000):
        Y = l1.forward_pass(X)
        print(i)