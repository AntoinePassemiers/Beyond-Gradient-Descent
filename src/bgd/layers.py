""" This module contains all the layers that are implemented.
Any layer needs to inherit from :class:`bgd.layers.Layer` and
to implement its abstract methods (:obj:`_forward`, :obj:`backward`
and :obj:`get_parameters`, even by returning None if layer is
non-parametric). """

# layers.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'FullyConnected', 'Activation', 'Convolutional2D', 'MaxPooling2D',
    'Dropout', 'GaussianNoise', 'Flatten', 'Lambda'
]

from abc import ABCMeta, abstractmethod
import numpy as np

from bgd.initializers import ZeroInitializer, GlorotUniformInitializer
from bgd.errors import NonLearnableLayerError
# pylint: disable=import-error,no-name-in-module
from bgd.conv import conv_2d_forward, \
                     conv_2d_backward, conv_2d_backward_weights
from bgd.max_pooling import max_pooling_2d_backward, max_pooling_2d_forward


class Layer(metaclass=ABCMeta):
    """ Base class for neural and non-neural layers.
    Each layer must implement methods '_forward' and
    '_backward': the output shape during forward pass
    must be equal to the input shape during backward pass,
    and vice versa.

    Args:
        copy (bool):
            Whether to copy layer output.
        save_input (bool):
            Whether to keep a reference to the input array..
        save_output (bool):
            Whether to keep a reference to the output array.

    Attributes:
        input_shape (tuple):
            Shape of the input array.
        current_input (:obj:`np.ndarray`):
            Reference to the current input array.
        current_output (:obj:`np.ndarray`):
            Reference to the current output array.
        propagate (bool):
            Whether to propagate the signal. This is usually
            set to False for the first layer of a sequential
            model for example. Indeed, the first layer has no
            neighbouring layer to transfer the signal to.
    """

    def __init__(self, copy=False, save_input=True, save_output=True):
        self.input_shape = None
        self.copy = copy
        self.current_input = None
        self.current_output = None
        self.save_output = save_output
        self.save_input = save_input
        self.propagate = True

    def activate_propagation(self):
        """ Activate signal propagation during backpropagation. """
        self.propagate = True

    def deactivate_propagation(self):
        """ Deactivate signal propagation during backpropagation. """
        self.propagate = False

    def learnable(self):
        """ Tells whether the layer has learnable parameters. """
        parameters = self.get_parameters()
        return not (parameters is None or len(parameters) == 0)

    def update_parameters(self, delta_fragments):
        """ Update parameters: base class has no learnable parameters.
        Subclasses that are learnable must override this method.

        Args:
            delta_fragments (tuple):
                Tuple of NumPy arrays. Each array is a parameter update vector
                and is used to update parameters using the following formula:
                params = params - delta_fragment.
                The tuple can have a size > 1 for convenience. For example,
                a convolutional neural layer has one array to update the filters
                weights and one array to update the biases:
                weights = weights - delta_fragments[0]
                biases  = biases  - delta_fragments[1]
        """
        del delta_fragments  # unused
        raise NonLearnableLayerError(
            "Cannot update parameters of a %s layer" % self.__class__.__name__)

    def forward(self, X):
        """ Wrapper method for method '_forward'. Given an input array,
        checks the input shape, computes the output and saves the output if needed.

        Args:
            X (:obj:`np.ndarray`):
                Input array
        """
        self.input_shape = X.shape
        if self.save_input: # Save input if needed
            self.current_input = X
        current_output = self._forward(X)
        if self.save_output: # Save output if needed
            self.current_output = current_output
            # If needed and if the output does not produce
            # a copy of the input , then the output is copied.
            if self.copy and np.may_share_memory(
                    X, current_output, np.core.multiarray.MAY_SHARE_BOUNDS):
                self.current_output = np.copy(current_output)
        return current_output

    def backward(self, *args, **kwargs):
        """ Wrapper method for method '_backward'. Given an input array,
        checks the array shape, update parameters and propagate the
        signal if needed (and if layer is learnable).

        Returns a tuple of size 2 where first element is the signal to
        propagate and the second element is the error gradient with respect
        to the parameters of the layer. If propagation is deactivated
        for current layer, then the signal is replaced by None.
        If layer is non-learnable, the gradient is replaced by None.
        """
        if self.propagate or self.learnable():
            out = self._backward(*args, **kwargs)
            if not isinstance(out, tuple):
                # If wrapped method does not return a gradient vector,
                # then replace it by None
                out = (out, None)
            return out
        else:
            # Propagation is deactivated -> signal   == None
            # Layer is non-learnable     -> gradient == None
            return (None, None)

    @abstractmethod
    def get_parameters(self):
        """ Retuns a tuple containing the parameters of the layer.
        If layer is non-learnable, None is returned instead.
        The tuple can have a size > 1 for convenience. For example,
        a convolutional neural layer has one array for the filters
        weights and one array for the biases.
        """
        pass

    @abstractmethod
    def _forward(self, X):
        """ Wrapped method for applying a forward pass on input X. """
        pass

    @abstractmethod
    def _backward(self, error, extra_info=None):
        """ Wrapped method for applying a backward pass on input X. """
        pass


class FullyConnected(Layer):
    """ Fully connected (dense) neural layer. Each output neuron is a
    weighted sum of its inputs with possibly a bias.

    Args:
        n_in (int):
            Number of input neurons.
        n_out (int):
            Number of output neurons.
        copy (bool):
            Whether to copy layer output.
        with_bias (bool):
            Whether add a bias to output neurons.
        dtype (type):
            Type of weights and biases.
        initializer (:class:`bgd.initializers.Initializer`):
            Initializer of the weights.
        bias_initializer (:class:`bgd.initializers.Initializer`):
            Initializer of the biases.

    Attributes:
        weights (:obj:`np.ndarray`):
            Matrix of weights.
        biases (:obj:`np.ndarray`):
            Vector of biases.
    """

    def __init__(self, n_in, n_out, copy=False, with_bias=True,
                 dtype=np.double, initializer=GlorotUniformInitializer(),
                 bias_initializer=ZeroInitializer()):
        Layer.__init__(self, copy=copy, save_output=False)
        self.with_bias = with_bias
        self.dtype = dtype
        self.n_in = n_in
        self.n_out = n_out
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.weights = self.initializer.initialize((self.n_in, self.n_out), dtype=self.dtype)
        if self.with_bias:
            self.biases = self.bias_initializer.initialize((1, self.n_out), dtype=self.dtype)
        else:
            self.biases = None

    def _forward(self, X):
        # Output: X * W + b
        return np.dot(X, self.weights) + self.biases

    def _backward(self, error, extra_info=None):
        gradient_weights = np.dot(self.current_input.T, error)
        try:
            l2_alpha = extra_info['l2_reg']
            if l2_alpha > 0:
                # Derivative of L2 regularization term
                gradient_weights += l2_alpha * self.weights
        except KeyError:
            pass
        gradient_bias = np.sum(error, axis=0, keepdims=True)
        if self.with_bias:
            gradients = (gradient_weights, gradient_bias)
        else:
            gradients = gradient_weights
        if self.propagate:
            signal = np.dot(error, self.weights.T)
        else:
            signal = None
        return (signal, gradients)

    def get_parameters(self):
        if self.with_bias:
            return (self.weights, self.biases)
        return (self.weights,)

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
            e = np.exp(X)
            out = e / np.sum(e, axis=1, keepdims=True)
        else:
            raise NotImplementedError()
        return out

    def _backward(self, error, extra_info=None):
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
        dilated_kernel_shape = 1 + dilations * (np.asarray(kernel_shape[1:-1])-1)  # F_H^\delta and F_W^\delta
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

        out_shape = Convolutional2D._get_output_shape(filters_shape, in_shape,
                self.strides, self.dilations)
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

        conv_2d_forward(self.out_buffer, X.astype(np.float32), self.filters,
                        self.biases, self.strides, self.dilations, self.with_bias,
                        self.n_jobs)

        self.n_instances = X.shape[0]
        return self.out_buffer[:X.shape[0], :, :, :]

    def _backward(self, error, extra_info=None):
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
        conv_2d_forward(weights_buffer,
            a.transpose((3, 1, 2, 0)).astype(np.float32),
            error.transpose((3, 1, 2, 0)).astype(np.float32),
            self.biases,
            self.dilations,
            self.strides,
            False,
            self.n_jobs
        )
        weights_buffer = weights_buffer.transpose((3, 1, 2, 0))
        if extra_info['l2_reg'] > 0:
            # Derivative of L2 regularization term
            self.in_buffer += extra_info['l2_reg'] * self.filters
        if self.propagate:
            conv_2d_backward(self.error_buffer[:self.n_instances],
                             error.astype(np.float32), self.filters,
                             self.strides, self.n_jobs
                            )
            signal = self.error_buffer[:self.n_instances, :, :, :]
            return (signal, (weights_buffer, db))
        return (None, (self.in_buffer, db))

    def get_parameters(self):
        return (self.filters, self.biases) if self.with_bias else (self.filters,)

    def update_parameters(self, delta_fragments):
        self.filters -= delta_fragments[0]
        if self.with_bias:
            self.biases -= delta_fragments[1]


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

    def _backward(self, error, extra_info=None):
        max_pooling_2d_backward(self.in_buffer, error, self.mask, self.pool_shape, self.strides)
        return self.in_buffer[:error.shape[0], :, :, :]

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

    def _backward(self, error, extra_info=None):
        assert self.active
        return self.mask * error

    def get_parameters(self):
        return None # Non-parametric layer


class GaussianNoise(Layer):

    def __init__(self, mean, stdv, clip=(0, 1), copy=False):
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

    def _backward(self, error, extra_info=None):
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

    def _backward(self, error, extra_info=None):
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

    def _backward(self, error, extra_info=None):
        return self.backward_op(error)

    def get_parameters(self):
        return None # Non-parametric layer
