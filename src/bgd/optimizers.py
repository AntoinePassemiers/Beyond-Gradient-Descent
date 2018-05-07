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
        delta2 = 0
        if self.momentum > 0:
            if self.previous_grad is not None:
                delta2 = self.momentum * self.previous_grad
            self.previous_grad = delta
        return delta + delta2


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
        return self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))


class LBFGS(Optimizer):

    class History:

        def __init__(self, m):
            self.padding = 0
            self.m = m
            self.values = list()
        
        def __getitem__(self, key):
            return self.values[key-self.padding]
        
        def __setitem__(self, key, value):
            if key-self.padding == len(self.values):
                self.values.append(value)
                if len(self.values) > self.m:
                    self.values = self.values[1:]
                    self.padding += 1
            elif key-self.padding < len(self.values):
                self.values[key-self.padding] = value
    
        def __len__(self):
            return len(self.values) + self.padding

    def __init__(self, m):
        self.m = m
        self.k = -1
        self.previous_grad = None
        self.y = LBFGS.History(self.m)
        self.s = LBFGS.History(self.m)
        self.alpha = LBFGS.History(self.m)

    def _update(self, grad):
        if self.k >= self.m:
            q = np.copy(grad)
            for i in range(self.k-1, self.k-self.m-1, -1):
                rho_i = 1. / np.dot(self.s[i], self.y[i])
                q -= self.alpha[i] * self.y[i]

            z = inner_product(self.y[self.k-1] \
                / np.dot(self.y[self.k-1], self.y[self.k-1]), self.s[self.k-1], q)

            for i in range(self.k-self.m, self.k, 1):
                rho_i = 1. / np.dot(self.s[i], self.y[i])
                beta = rho_i * np.dot(self.y[i], z)
                z += self.s[i] * (self.alpha[i] - beta)
        else:
            z = grad

        # Update history
        if self.old_grad is not None:
            self.y[self.k] = grad - self.old_grad
            self.s[self.k] = ss
            rho_i = 1. / np.dot(self.s[self.k], self.y[self.k])
            self.alpha[self.k] = rho_i * np.dot(self.s[self.k], grad)

        self.old_grad = grad
        self.k += 1

        # TODO: LINE SEARCH
        aa = .001

        return aa * z