# -*- coding: utf-8 -*-
# errors.py
# author : Antoine Passemiers, Robin Petit

from abc import ABCMeta, abstractmethod
import numpy as np


class Error(metaclass=ABCMeta):
    """ Base class for error operators.

    Args:
        X (np.ndarray):
            Array of shape (n_samples, output_size) representing
            the predictions. Must have same length as Y.
        Y (np.ndarray):
            Array of shape (n_samples, output_size) representing
            the expected outputs. Must have same length as X.
    """

    def eval(self, y, y_hat):
        """ Evaluates the error operator and returns an array of
        shape (n_samples,) with the per-sample cost.
        """
        assert len(y) == len(y_hat)
        return self._eval(y, y_hat)

    def grad(self, y, y_hat):
        """ Evaluates the gradient of the error. The error must be
        inconditional to the model parameters. In other words, the
        error is non-parametric.
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


class MSE(Error):
    """ Differentiable Mean Squared Error operator """

    def _eval(self, y, y_hat):
        """ Return mean squared error.

        Args:
            y (np.ndarray): ground truth values
            y_hat (np.ndarray): predicted values
        """
        return np.mean(.5 * (y_hat - y) ** 2)

    def _grad(self, y, y_hat):
        """ Return derivative of mean squared error.

        Args:
            y (np.ndarray): ground truth values
            y_hat (np.ndarray): predicted values
        """
        return y_hat - y


class CrossEntropy(Error):
    """ Differentiable cross-entropy operator.

    Attributes:
        epsilon: parameter for numerical stability
    """

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def _eval(self, y, y_hat):
        """ Return cross-entropy metric.

        Args:
            y (np.ndarray): ground truth labels
            y_hat (np.ndarray): predicted values
        """
        indices = np.argmax(y, axis=1).astype(np.int)
        predictions = y_hat[np.arange(len(y_hat)), indices]
        log_predictions = np.log(np.maximum(predictions, self.epsilon))
        return -np.mean(log_predictions)

    def _grad(self, y, y_hat):
        """ Return derivative of cross-entropy function.

        Args:
            y (np.ndarray): ground truth labels
            y_hat (np.ndarray): predicted values
        """
        return y_hat - y
