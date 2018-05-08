# -*- coding: utf-8 -*-
# optimizers.py
# author : Antoine Passemiers, Robin Petit

from abc import ABCMeta, abstractmethod
import numpy as np


class Optimizer(metaclass=ABCMeta):

    @abstractmethod
    def _update(self, grad):
        pass
    
    def update(self, grad):
        in_shape = grad.shape
        delta = self._update(grad.flatten(order='C'))
        return delta.reshape(in_shape, order='C')


class MomentumOptimizer(Optimizer):

    def __init__(self, learning_rate=.005, momentum=.9):
        assert(0 <= momentum <= 1)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.previous_grad = None
    
    def _update(self, grad):
        delta = self.learning_rate * grad
        if self.momentum > 0:
            if self.previous_grad is not None:
                delta += self.momentum * self.previous_grad
            self.previous_grad = delta
        return delta


class AdamOptimizer(Optimizer):

    def __init__(self, learning_rate=.001, beta_1=.9, beta_2=.999, epsilon=1e-8):
        """
        References
        ----------
        ADAM: A Method For Stochastic Optimization
        Diederik P. Kingma and Jimmy Lei Ba
        https://arxiv.org/pdf/1412.6980.pdf
        """
        assert((0 <= beta_1 < 1) and (0 <= beta_2 < 1))
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.step = 0
        self.moment_1 = 0
        self.moment_2 = 0
    
    def _update(self, grad):
        self.step += 1
        self.moment_1 = self.beta_1 * self.moment_1 + (1. - self.beta_1) * grad
        self.moment_2 = self.beta_2 * self.moment_2 + (1. - self.beta_2) * grad ** 2
        m_hat = self.moment_1 / (1. - self.beta_1 ** self.step)
        v_hat = self.moment_2 / (1. - self.beta_2 ** self.step)
        delta = self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))
        return delta


class LBFGS(Optimizer):

    def __init__(self, m=10):
        self.m = m
        self.k = -1
        self.previous_grad = None
        self.y = list()
        self.s = list()
        self.alpha = list()

    def _update(self, grad):
        #print(batch_grad.shape)
        if self.k >= self.m:
            q = np.copy(grad)
            for s_i, y_i, alpha_i in reversed(list(zip(self.s, self.y, self.alpha))):
                q -= alpha_i * y_i

            z = self.inner_product(self.y[-1] \
                / np.dot(self.y[-1], self.y[-1]), self.s[-1], q)

            for s_i, y_i, alpha_i in zip(self.s, self.y, self.alpha):
                rho_i = 1. / np.dot(s_i, y_i)
                beta_i = rho_i * np.dot(y_i, z)
                z += s_i * (alpha_i - beta_i)
        else:
            z = grad

        # TODO: LINE SEARCH
        steplength = self.compute_steplength(grad)
        steplength = .005
        delta = steplength * z

        # Update history
        if self.previous_grad is not None:
            self.y.append(grad - self.previous_grad)
            self.s.append(delta)
            rho_i = 1. / np.dot(self.s[-1], self.y[-1])
            self.alpha.append(rho_i * np.dot(self.s[-1], grad))

            if len(self.y) > self.m:
                self.y = self.y[1:]
                self.s = self.s[1:]
                self.alpha = self.alpha[1:]

        self.previous_grad = grad
        self.k += 1

        return delta

    def inner_product(self, a, b, c):
        return np.repeat(np.dot(a, c), len(a)) * b
    
    def compute_steplength(self, grad):
        # print(grad.shape)
        pass