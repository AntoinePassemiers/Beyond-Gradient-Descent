# -*- coding: utf-8 -*-
# errors.py
# author : Antoine Passemiers, Robin Petit

from abc import ABCMeta, abstractmethod
import numpy as np


class Error(metaclass=ABCMeta):
    
    @abstractmethod
    def eval(self, X, Y):
        pass
    
    @abstractmethod
    def grad(self, X, Y):
        pass


class MSE(Error):

    def eval(self, y, y_hat):
        return (.5 * (y_hat - y) ** 2) / len(y)
    
    def grad(self, y, y_hat):
        return y_hat - y


class CrossEntropy(Error):

    def eval(self, y, probs):
        indices = np.argmax(y, axis = 1).astype(np.int)
        log_predictions = np.log(probs[np.arange(len(probs)), indices])
        return -np.sum(log_predictions) / len(log_predictions)
    
    def grad(self, y, probs):
        return probs - y