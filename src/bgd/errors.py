# -*- coding: utf-8 -*-
# errors.py
# author : Antoine Passemiers, Robin Petit

from abc import ABCMeta, abstractmethod
import numpy as np


class Error(metaclass=ABCMeta):

    def eval(self, X, Y):
        assert(len(X) == len(Y))
        return self._eval(X, Y)

    @abstractmethod
    def _eval(self, X, Y):
        pass

    @abstractmethod
    def grad(self, X, Y):
        pass


class MSE(Error):

    def _eval(self, y, y_hat):
        return np.mean(.5 * (y_hat - y) ** 2)

    def grad(self, y, y_hat):
        return y_hat - y


class CrossEntropy(Error):
    """ Differentiable cross-entropy operator.

    Attributes:
        epsilon: parameter for numerical stability
    """

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def _eval(self, y, probs):
        indices = np.argmax(y, axis=1).astype(np.int)
        predictions = probs[np.arange(len(probs)), indices]
        log_predictions = np.log(np.maximum(predictions, self.epsilon))
        return -np.mean(log_predictions)

    def grad(self, y, probs):
        return probs - y
