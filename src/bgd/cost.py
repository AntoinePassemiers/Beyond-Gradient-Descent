""" This module contains all the cost (loss) functions that
are implemented. There exists a separation between regression
and classification classes.

Each cost function implemented needs to implement the
computation of the cost (obviously) and *also* the gradient
of the function w.r.t. :obj:`y_hat`.

Any new cost function needs to inherit from either
:class:`bgd.cost.RegressionCost` or :class:`ClassificationCost`
depending on its use. """

# cost.py
# author : Antoine Passemiers, Robin Petit

__all__ = ['MSE', 'CrossEntropy']

from abc import ABCMeta, abstractmethod
import sys

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

    def print_fitness(self, y, y_hat, end='\n', files=sys.stdout):
        r""" Prints a measure of the fitness/cost.

        Args:
            y (:class:`numpy.ndarray`):
                Ground truth.
            y_hat (:class:`numpy.ndarray`):
                Prediction of the model.
            end (:class:`str`):
                String to write at the end. Defaults to '\n'.
            files (:class:`tuple` of file-like objects or file-like object):
                File(s) to write the fitness in. Defaults to stdout.
        """
        if not isinstance(files, (list, tuple, set)):
            files = (files,)
        for f in files:
            self._print_fitness(y, y_hat, end, f)

    @abstractmethod
    def _print_fitness(self, y, y_hat, end, f):
        """ Wrapped method for printing the cost value/accuracy
        of the model on provided data.

        Args:
            see :meth:`print_fitness`.
        """
        pass

    @abstractmethod
    def name(self):
        """ Returns the name of the cost function. """
        pass


class ClassificationCost(Cost):  #pylint: disable=W0223
    """ Base class for all the cost functions used in classification. """
    def _print_fitness(self, y, y_hat, end, f):
        f.write('accuracy: {:.3f}%{}' \
                .format(100*ClassificationCost.accuracy(y, y_hat), end))

    @staticmethod
    def accuracy(y, y_hat):
        """ Returns the accuracy of y_hat w.r.t. y. """
        return (y_hat.argmax(axis=1) == y).sum() / len(y_hat)


class RegressionCost(Cost):  #pylint: disable=W0223
    """ Base class for all the cost functions used in regression. """
    def _print_fitness(self, y, y_hat, end, f):
        f.write('{}: {:.3f}{}' \
                .format(self.name(), self.eval(y, y_hat), end))


class MSE(RegressionCost):
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
        if len(y.shape) < len(y_hat.shape):
            y = y[..., np.newaxis]
        return y_hat - y

    def name(self):
        return 'MSE'


class CrossEntropy(ClassificationCost):
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

    def name(self):
        return 'cross-entropy'
