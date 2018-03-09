# -*- coding: utf-8 -*-
# optimizers.py
# author : Antoine Passemiers, Robin Petit

from abc import ABCMeta, abstractmethod
import numpy as np


class Optimizer(metaclass=ABCMeta):

    @abstractmethod
    def update(self, grad):
        pass


class MomentumOptimizer(Optimizer):

    def __init__(self, learning_rate=.005, momentum=.9):
        assert(0 <= momentum <= 1)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.previous_grad = None
    
    def update(self, grad):
        # TODO: divide learning rate by batch size ?
        delta = self.learning_rate * grad
        delta2 = 0
        if self.momentum > 0:
            if self.previous_grad is not None:
                delta2 = self.momentum * self.previous_grad
            self.previous_grad = delta
        return delta + delta2


class AdamOptimizer(Optimizer):

    def __init__(self, learning_rate=.001, beta_1=.9, beta_2=.999, epsilon=1e-08):
        """
        References
        ----------
        ADAM: A Method For Stochastic Optimization
        Diederik P. Kingma and Jimmy Lei Ba
        https://arxiv.org/pdf/1412.6980.pdf
        """
        assert((0 < beta_1 < 1) and (0 < beta_2 < 1))
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.step = 0
        self.moment_1 = 0
        self.moment_2 = 0
    
    def update(self, grad):
        self.step += 1
        self.moment_1 = self.beta_1 * self.moment_1 + (1. - self.beta_1) * grad
        self.moment_2 = self.beta_2 * self.moment_2 + (1. - self.beta_2) * grad ** 2
        m_hat = self.moment_1 / (1. - self.beta_1 ** self.step)
        v_hat = self.moment_2 / (1. - self.beta_2 ** self.step)
        return self.learning_rate * (m_hat / (np.sqrt(v_hat) - self.epsilon))