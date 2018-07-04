# cost.py
# author : Antoine Passemiers, Robin Petit

from abc import ABCMeta, abstractmethod
import numpy as np


class Cost(metaclass=ABCMeta):
    """ Base class for error operators.

    Args:
        y (:obj:`np.ndarray`):
            Array of shape (n_samples, output_size) representing
            the expected outputs. Must have same length as X.
        y_hat (:obj:`np.ndarray`):
            Array of shape (n_samples, output_size) representing
            the predictions. Must have same length as Y.
    """

    def eval(self, y, y_hat):
        """ Evaluates the error operator and returns an array of
        shape (n_samples,) with the per-sample cost.

        Returns:
            :obj:`np.ndarray`:
                loss:
                    The error function value for each sample
                    of the batch. `shape == (len(y),)`
        """
        assert len(y) == len(y_hat)
        return self._eval(y, y_hat)

    def grad(self, y, y_hat):
        """ Evaluates the gradient of the error. The error must be
        not depend on to the model parameters.

        Returns:
            :obj:`np.ndarray`:
                grad:
                    The gradient of the error for each sample
                    of the batch. `shape == (len(y),)`
        """
        assert len(y) == len(y_hat)
        return self._grad(y, y_hat)

    @abstractmethod
    def _eval(self, y, y_hat):
        """ Wrapped method for evaluating the error operator.
        Subclasses must override this method. """
        pass

    @abstractmethod
    def _grad(self, y, y_hat):
        """ Wrapped method for evaluating the error gradient.
        Subclasses must override this method. """
        pass


class MSE(Cost):
    r""" Differentiable Mean Squared Error operator.

    .. math::
        C(y, \hat y) = \frac 1{2n}\sum_{i=1}^n\big(y_i - \hat y_i\big)^2
    """

    def _eval(self, y, y_hat):
        """ Return mean squared error.

        Args:
            y (:obj:`np.ndarray`):
                Ground truth values.
            y_hat (:obj:`np.ndarray`):
                Predicted values.

        Returns:
            :obj:`np.ndarray`:
                loss:
                    The error function value for each sample
                    of the batch. `shape == (len(y),)`
        """
        return .5 * np.mean((y_hat - y) ** 2)

    def _grad(self, y, y_hat):
        """ Returns the derivative of mean squared error.

        Args:
            y (:obj:`np.ndarray`):
                Ground truth values.
            y_hat (:obj:`np.ndarray`):
                Predicted values.

        Returns:
            :obj:`np.ndarray`:
                grad:
                    The gradient of the error for each sample
                    of the batch. `shape == (len(y),)`
        """
        return y_hat - y


class CrossEntropy(Cost):
    r""" Differentiable cross-entropy operator.

    .. math::
        C(y, \hat y) = -\frac 1n\sum_{i=1}^n\big[y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)\big]

    Attributes:
        epsilon: parameter for numerical stability
    """

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def _eval(self, y, y_hat):
        """ Return cross-entropy metric.

        Args:
            y (:obj:`np.ndarray`):
                Ground truth labels.
            y_hat (:obj:`np.ndarray`):
                Predicted values.

        Returns:
            :obj:`np.ndarray`:
                loss:
                    The error function value for each sample
                    of the batch. `shape == (len(y),)`
        """
        indices = np.argmax(y, axis=1).astype(np.int)
        predictions = y_hat[np.arange(len(y_hat)), indices]
        log_predictions = np.log(np.maximum(predictions, self.epsilon))
        return -np.mean(log_predictions)

    def _grad(self, y, y_hat):
        """ Return derivative of cross-entropy function.

        Args:
            y (:obj:`np.ndarray`):
                Ground truth labels.
            y_hat (:obj:`np.ndarray`):
                Predicted values.

        Returns:
            :obj:`np.ndarray`:
                grad:
                    The gradient of the error for each sample
                    of the batch. `shape == (len(y),)`
        """
        return y_hat - y
