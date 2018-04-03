# -*- coding: utf-8 -*-
# layers.py
# author : Antoine Passemiers, Robin Petit

from bgd.initializers import GaussianInitializer
from bgd.operators import *

from abc import ABCMeta, abstractmethod
import numpy as np

class Layer(metaclass=ABCMeta):

    def __init__(self, copy=False):
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
        ret = np.dot(error, self.weights.T)
        self.update(gradient_weights, gradient_bias, extra_info['optimizer'])
        return ret

    def update(self, dW, db, optimizer):
        self.weights -= optimizer.update(dW)
        if self.with_bias:
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
            grad_X = self.current_input
            if self.copy:
                grad_X = np.empty_like(grad_X)
            grad_X[:] = (grad_X >= 0)
        elif self.function == 'softmax':
            grad_X = X * (1. - X)
        else:
            raise NotImplementedError()
        return grad_X * error

    def get_parameters(self):
        return None


class Convolutional2D(Layer):

    def __init__(self, filter_shape, n_filters, strides=[1, 1], with_bias=True, copy=True, initializer=GaussianInitializer(0, .02)):
        Layer.__init__(self, copy=copy)
        self.filter_shape = filter_shape  # [height, width, n_channels]
        self.strides = strides
        self.initializer = initializer
        self.with_bias = with_bias
        self.n_filters = n_filters
        assert(len(filter_shape) == 3)
        self.filters = None
        self.biases = None
        self.in_buffer = None
        self.out_buffer = None

    def init_weights(self, dtype, in_shape):
        filters_shape = tuple([self.n_filters] + list(self.filter_shape))
        self.filters = self.initializer.initialize(filters_shape, dtype=dtype)
        self.biases = self.initializer.initialize(self.n_filters, dtype=dtype)  #np.zeros(self.n_filters, dtype=dtype)

        out_height = (in_shape[1] - (self.filter_shape[0]-1)) // self.strides[0]
        out_width = (in_shape[2] - (self.filter_shape[1]-1)) // self.strides[1]
        out_shape = (in_shape[0], out_height, out_width, self.n_filters)
        self.out_buffer = np.empty(out_shape, dtype=dtype)
        self.in_buffer = np.empty(self.filters.shape, dtype=dtype)
        self.error_buffer = np.empty(in_shape, dtype=dtype)

    def _forward(self, X):
        if X.ndim == 3:
            X = X[..., np.newaxis]
        if self.filters is None:
            self.init_weights(X.dtype, X.shape)
        # TODO: apply convolution with numpy or cython ?
        if X.shape[0] > self.out_buffer.shape[0]:
            self.out_buffer = np.empty([X.shape[0]] + list(self.out_buffer.shape)[1:])
        conv_2d_forward(self.out_buffer, X, self.filters, self.biases, self.strides, self.with_bias)
        return self.out_buffer[:X.shape[0], :, :, :]

    def _backward(self, error, extra_info):
        db = np.sum(error, axis=(0, 1, 2))  # sum on 3 first dimensions to only keep the 4th (i.e. n_filters)
        db /= error.shape[0]
        if self.current_input.ndim == 3:
            a = self.current_input[..., np.newaxis]
        else:
            a = self.current_input
        conv_2d_backward_weights(self.in_buffer, a, error, self.strides)
        self.in_buffer /= error.shape[0]
        if extra_info['l2_reg'] > 0:
            self.in_buffer += extra_info['l2_reg'] * self.filters  # Derivative of L2 regularization term
        conv_2d_backward(self.error_buffer, error, self.filters, self.strides)  ## uncomment to compute error to propagate
        self.update(self.in_buffer, db, extra_info['optimizer'])
        return self.error_buffer

    def update(self, dW, db, optimizer):
        self.filters -= optimizer.update(dW)
        if self.with_bias:
            self.biases -= optimizer.learning_rate * db

    def get_parameters(self):
        return (self.filters, self.biases) if self.with_bias else (self.filters,)


class MaxPooling2D(Layer):

    def __init__(self, pool_shape, strides=[1, 1], copy=True):
        Layer.__init__(self, copy=copy)
        self.pool_shape = pool_shape
        self.strides = strides
        self.mask = None
        self.out_buffer = None
        self.in_buffer = None

    def _forward(self, X):
        if self.out_buffer is None or X.shape[0] > self.out_buffer.shape[0]:
            out_height = (X.shape[1] - self.pool_shape[0] + 1) // self.strides[0]
            out_width = (X.shape[2] - self.pool_shape[1] + 1) // self.strides[1]
            self.out_buffer = np.empty((X.shape[0], out_height, out_width, X.shape[3]), dtype=X.dtype)
            self.in_buffer = np.empty_like(X)
            self.mask = np.empty(X.shape, dtype=np.int8)
        max_pooling_2d_forward(self.out_buffer, self.mask, X, self.pool_shape, self.strides)
        return self.out_buffer[:X.shape[0], :, :, :]

    def _backward(self, error, extra_info={}):
        max_pooling_2d_backward(self.in_buffer, error, self.mask, self.pool_shape, self.strides)
        return self.in_buffer[:error.shape[0], :, :, :]

    def get_parameters(self):
        return None


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
        return self.backward_op(error)

    def get_parameters(self):
        return None
