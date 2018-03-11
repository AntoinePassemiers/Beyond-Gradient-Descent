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

    def __init__(self, n_in, n_out, copy=True, with_bias=True, dtype=np.double, initializer=GaussianInitializer(0, .01)):
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

    def _forward(self, X):
        return np.dot(X, self.weights) + self.biases
    
    def _backward(self, error, extra_info={}):
        batch_size = len(error)
        gradient_weights = np.dot(self.current_input.T, error) / batch_size
        if extra_info['l2_reg'] > 0:
            gradient_weights += extra_info['l2_reg'] * self.weights # Derivative of L2 regularization term
        gradient_bias = np.sum(error, axis=0, keepdims=True) / batch_size
        self.update(gradient_weights, gradient_bias, extra_info['optimizer'])
        return np.dot(error, self.weights.T)

    def update(self, dW, db, optimizer):
        self.weights -= optimizer.update(dW)
        if self.with_bias:
            # TODO: Apply optimization on biases as well
            self.biases -= optimizer.learning_rate * db

    def get_parameters(self):
        return (self.weights, self.biases) if self.with_bias else (self.weights,)


class Activation(Layer):

    def __init__(self, function='sigmoid', copy=True):
        Layer.__init__(self, copy=copy)
        self.function = function.lower()
        self.copy = copy

    def _forward(self, X):
        if self.function == 'sigmoid':
            out = 1. / (1. + np.exp(-X))
        elif self.function == 'tanh':
            out = np.tanh(X)
        elif self.function == 'relu':
            out = np.maximum(X, 0)
        elif self.function == 'softmax':
            e = np.exp(X)
            out = e / np.sum(e, axis=1, keepdims=True)
        else:
            raise NotImplementedError()
        return out
    
    def _backward(self, error, extra_info={}):
        X = self.current_output
        if self.function == 'sigmoid':
            grad_X = X * (1. - X)
        elif self.function == 'tanh':
            grad_X = 1. - X ** 2
        elif self.function == 'relu':
            grad_X = X
            if self.copy:
                grad_X = np.empty_like(X)
            grad_X[:] = (X >= 0)
        elif self.function == 'softmax':
            grad_X = X * (1. - X)
        else:
            raise NotImplementedError()
        return grad_X * error

    def get_parameters(self):
        return None


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

    def get_parameters(self):
        return (self.filters, self.biases) if self.with_bias else (self.filters,)


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
    
    def _forward(self, X):
        if self.active:
            self.mask = (np.random.rand(*X.shape) > (1. - self.keep_proba))
            return self.mask * X
        else:
            return X
    
    def _backward(self, error, extra_info={}):
        assert(self.active)
        return self.mask * error
    
    def get_parameters(self):
        return None


class Flatten(Layer):

    def __init__(self, order='C', copy=True):
        Layer.__init__(self, copy=copy)
        self.order = order
        self.in_shape = None
    
    def _forward(self, X):
        self.in_shape = X.shape
        return X.reshape((X.shape[0], -1), order=self.order)
    
    def _backward(self, error, extra_info={}):
        return error.reshape(self.in_shape, order=self.order)

    def get_parameters(self):
        return None


class GaussianNoise(Layer):

    def __init__(self, stdv, clip=(0, 1), copy=True):
        Layer.__init__(self, copy=copy)
        self.stdv = stdv
        self.clip = clip
    
    def _forward(self, X):
        noised_X = X + np.random.normal(0, self.stdv)
        if self.clip:
            noised_X = np.clip(noised_X, self.clip[0], self.clip[1])
        return noised_X
    
    def _backward(self, error, extra_info={}):
        return error

    def get_parameters(self):
        return None


class Lambda(Layer):

    def __init__(self, forward_op, backward_op, copy=True):
        Layer.__init__(self, copy=copy)
        self.forward_op = forward_op
        self.backward_op = backward_op
    
    def _forward(self, X):
        return self.forward_op(X)
    
    def _backward(self, error, extra_info={}):
        return self.backward_op(X)
    
    def get_parameters(self):
        return None
