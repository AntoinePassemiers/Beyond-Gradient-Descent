# layers/conv2d.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'Convolutional2D'
]

import numpy as np

from bgd.initializers import ZeroInitializer, GlorotUniformInitializer
from .layer import Layer
# pylint: disable=import-error,no-name-in-module
from .conv import conv_2d_forward, conv_2d_backward

class Convolutional2D(Layer):

    def __init__(self, filter_shape, n_filters, strides=(1, 1),
                 dilations=(1, 1), with_bias=True, copy=False,
                 initializer=GlorotUniformInitializer(),
                 bias_initializer=ZeroInitializer(), n_jobs=4):
        Layer.__init__(self, copy=copy, save_output=False)
        if len(filter_shape) != 3:
            raise ValueError('Wrong shape for filters!')
        self.filter_shape = np.asarray(filter_shape, dtype=np.int)  # [height, width, n_channels]
        self.strides = np.asarray(strides, dtype=np.int)
        self.dilations = np.asarray(dilations, dtype=np.int)
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.with_bias = with_bias
        self.n_filters = n_filters
        self.filters = None
        self.biases = None
        self.in_buffer = None
        self.out_buffer = None
        self.n_jobs = n_jobs
        self.error_buffer = None
        self.n_instances = -1

    @staticmethod
    def _get_output_shape(kernel_shape, input_shape, strides, dilations):
        # F_H^\delta and F_W^\delta
        dilated_kernel_shape = 1 + dilations * (np.asarray(kernel_shape[1:-1])-1)
        return [
            input_shape[0],
            1 + (input_shape[1] - dilated_kernel_shape[0]) // strides[0],
            1 + (input_shape[2] - dilated_kernel_shape[1]) // strides[1],
            kernel_shape[0],
        ]

    def init_weights(self, dtype, in_shape):
        filters_shape = tuple([self.n_filters] + list(self.filter_shape))
        self.filters = self.initializer.initialize(filters_shape, dtype=dtype)
        self.biases = self.bias_initializer.initialize(self.n_filters, dtype=dtype)

        out_shape = Convolutional2D._get_output_shape(
            filters_shape, in_shape, self.strides, self.dilations)
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
            self.out_buffer = np.empty(new_shape, dtype=np.float32)

        conv_2d_forward(self.out_buffer, X, self.filters,
                        self.biases, self.strides, self.dilations, self.with_bias,
                        self.n_jobs)

        self.n_instances = X.shape[0]
        return self.out_buffer[:X.shape[0], :, :, :]

    def _backward(self, error):
        # sum on 3 first dimensions to only keep the 4th (i.e. n_filters)
        db = np.sum(error, axis=(0, 1, 2))
        if self.current_input.ndim == 3:
            a = self.current_input[..., np.newaxis]
        else:
            a = self.current_input

        delta_shape = Convolutional2D._get_output_shape(
            error.transpose((3, 1, 2, 0)).shape,
            a.transpose((3, 1, 2, 0)).shape,
            self.dilations, self.strides
        )
        weights_buffer = np.empty(delta_shape, dtype=np.float32)
        conv_2d_forward(
            weights_buffer,
            a.transpose((3, 1, 2, 0)),
            error.transpose((3, 1, 2, 0)),
            self.biases,
            self.dilations,
            self.strides,
            False,
            self.n_jobs
        )
        weights_buffer = weights_buffer.transpose((3, 1, 2, 0))
        if self.propagate:
            conv_2d_backward(self.error_buffer[:self.n_instances],
                             error, self.filters, self.strides, self.n_jobs)
            signal = self.error_buffer[:self.n_instances, :, :, :]
            return (signal, (weights_buffer, db))
        return (None, (weights_buffer, db))

    def get_parameters(self):
        return (self.filters, self.biases) if self.with_bias else (self.filters,)

    def update_parameters(self, delta_fragments):
        self.filters -= delta_fragments[0]
        if self.with_bias:
            self.biases -= delta_fragments[1]
