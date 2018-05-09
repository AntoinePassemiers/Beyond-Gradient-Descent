# -*- coding: utf-8 -*-
# layers.py
# author : Antoine Passemiers, Robin Petit

from bgd.initializers import GaussianInitializer
from bgd.operators import *

import copy
from abc import ABCMeta, abstractmethod
import numpy as np

class Layer(metaclass=ABCMeta):

    def __init__(self, copy=False, save_input=True, save_output=True):
        self.input_shape = None
        self.copy = copy
        self.current_input = None
        self.current_output = None
        self.with_bias = False
        self.save_output = save_output
        self.save_input = save_input
        self.propagate = True
    
    def activate_propagation(self):
        self.propagate = True

    def deactivate_propagation(self):
        self.propagate = False
    
    def learnable(self):
        parameters = self.get_parameters()
        return not (parameters is None or len(parameters) == 0)

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
        self.input_shape = X.shape
        if self.save_input:
            self.current_input = X
        current_output = self._forward(X)
        if self.save_output:
            self.current_output = current_output
            if self.copy and np.may_share_memory(X, current_output, np.core.multiarray.MAY_SHARE_BOUNDS):
                self.current_output = np.copy(current_output)
        return current_output

    def backward(self, *args, **kwargs):
        if self.propagate or self.learnable():
            out = self._backward(*args, **kwargs)
            if not isinstance(out, tuple):
                out = (out, None)
            (signal, gradient) = out
            assert(signal.shape == self.input_shape)
            return out
        else:
            return (None, None)

    def update_parameters(self, delta_fragments):
        raise NonLearnableLayerError(
            "Cannot update parameters of a %s layer" % self.__class__.__name__)


class FullyConnected(Layer):

    def __init__(self, n_in, n_out, copy=False, with_bias=True,
                 dtype=np.double, initializer=GaussianInitializer(0, .01)):
        Layer.__init__(self, copy=copy, save_output=False)
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
        gradient_weights = np.dot(self.current_input.T, error)
        if extra_info['l2_reg'] > 0:
            gradient_weights += extra_info['l2_reg'] * self.weights # Derivative of L2 regularization term
        gradient_bias = np.sum(error, axis=0, keepdims=True)
        gradients = (gradient_weights, gradient_bias) if self.with_bias else gradient_weights
        if self.propagate:
            signal = np.dot(error, self.weights.T)
            return (signal, gradients)
        else:
            return (None, gradients)

    def get_parameters(self):
        return (self.weights, self.biases) if self.with_bias else (self.weights,)
    
    def update_parameters(self, delta_fragments):
        self.weights -= delta_fragments[0]
        if self.with_bias:
            self.biases -= delta_fragments[1]


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
            e = np.nan_to_num(np.exp(X))
            out = e / np.sum(e, axis=1, keepdims=True)
        else:
            raise NotImplementedError()
        return out

    def _backward(self, error, extra_info={}):
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
        return None


class Convolutional2D(Layer):

    def __init__(self, filter_shape, n_filters, strides=[1, 1], with_bias=True,
                 copy=False, initializer=GaussianInitializer(0, .02), n_jobs=4):
        Layer.__init__(self, copy=copy, save_output=False)
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
        self.n_jobs = n_jobs

    def init_weights(self, dtype, in_shape):
        filters_shape = tuple([self.n_filters] + list(self.filter_shape))
        self.filters = self.initializer.initialize(filters_shape, dtype=dtype)
        self.biases = self.initializer.initialize(self.n_filters, dtype=dtype)

        out_height = (in_shape[1] - (self.filter_shape[0]-1)) // self.strides[0]
        out_width = (in_shape[2] - (self.filter_shape[1]-1)) // self.strides[1]
        out_shape = (in_shape[0], out_height, out_width, self.n_filters)
        self.out_buffer = np.zeros(out_shape, dtype=dtype)
        self.in_buffer = np.zeros(self.filters.shape, dtype=dtype)
        self.error_buffer = np.zeros(in_shape, dtype=dtype)

    def _forward(self, X):
        if X.ndim == 3:
            X = X[..., np.newaxis]
        if self.filters is None:
            self.init_weights(np.float32, X.shape)
        if X.shape[0] > self.out_buffer.shape[0]:
            new_shape = tuple([X.shape[0]] + list(self.out_buffer.shape)[1:])
            self.out_buffer = np.empty(new_shape)

        conv_2d_forward(self.out_buffer, X.astype(np.float32), self.filters,
                        self.biases, self.strides, self.with_bias, self.n_jobs)
        G = np.copy(self.out_buffer)
        conv_2d_forward_sse(self.out_buffer, X.astype(np.float32), self.filters,
                            self.biases, self.strides, self.with_bias)
        G2 = np.copy(self.out_buffer)
        print("Forward - similarity between regular version and SSE version: %f" \
            % (np.isclose(G, G2).sum() / float(G.size)))
        
        self.n_instances = X.shape[0]
        return self.out_buffer[:X.shape[0], :, :, :]

    def _backward(self, error, extra_info):
        db = np.sum(error, axis=(0, 1, 2))  # sum on 3 first dimensions to only keep the 4th (i.e. n_filters)
        if self.current_input.ndim == 3:
            a = self.current_input[..., np.newaxis]
        else:
            a = self.current_input

        conv_2d_backward_weights(self.in_buffer, a.astype(np.float32),
                                 error.astype(np.float32), self.strides, self.n_jobs)
        if extra_info['l2_reg'] > 0:
            self.in_buffer += extra_info['l2_reg'] * self.filters  # Derivative of L2 regularization term
        if self.propagate:
            conv_2d_backward(self.error_buffer[:self.n_instances],
                             error.astype(np.float32), self.filters, self.strides, self.n_jobs)
            signal = self.error_buffer[:self.n_instances, :, :, :]
            return (signal, (self.in_buffer, db))
        else:
            return (None, (self.in_buffer, db))

    def get_parameters(self):
        return (self.filters, self.biases) if self.with_bias else (self.filters,)

    def update_parameters(self, delta_fragments):
        self.filters -= delta_fragments[0]
        if self.with_bias:
            self.biases -= delta_fragments[1]


class MaxPooling2D(Layer):

    def __init__(self, pool_shape, strides=[1, 1], copy=False):
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
            self.out_buffer = np.empty((X.shape[0], out_height, out_width, X.shape[3]), dtype=X.dtype)
            self.in_buffer = np.empty(X.shape, dtype=X.dtype)
            self.mask = np.empty(X.shape, dtype=np.int8)
        max_pooling_2d_forward(self.out_buffer, self.mask, X, self.pool_shape, self.strides)
        return self.out_buffer[:X.shape[0], :, :, :]

    def _backward(self, error, extra_info={}):
        max_pooling_2d_backward(self.in_buffer, error, self.mask, self.pool_shape, self.strides)
        return self.in_buffer[:error.shape[0], :, :, :]

    def get_parameters(self):
        return None


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
        else:
            return X

    def _backward(self, error, extra_info={}):
        assert(self.active)
        return self.mask * error

    def get_parameters(self):
        return None


class Flatten(Layer):

    def __init__(self, order='C', copy=False):
        Layer.__init__(self, copy=copy, save_input=False, save_output=False)
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

    def __init__(self, stdv, clip=(0, 1), copy=False):
        Layer.__init__(self, copy=copy, save_input=False, save_output=False)
        self.save_input = False
        self.save_output = False
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

    def __init__(self, forward_op, backward_op, copy=False):
        Layer.__init__(self, copy=copy)
        self.forward_op = forward_op
        self.backward_op = backward_op

    def _forward(self, X):
        return self.forward_op(X)

    def _backward(self, error, extra_info={}):
        return self.backward_op(error)

    def get_parameters(self):
        return None
