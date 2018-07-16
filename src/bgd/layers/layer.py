# layers/layer.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'Layer'
]

from abc import ABCMeta, abstractmethod
import numpy as np

from bgd.errors import NonLearnableLayerError

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
    def _backward(self, error):
        """ Wrapped method for applying a backward pass on input X. """
        pass
