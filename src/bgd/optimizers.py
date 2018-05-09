# -*- coding: utf-8 -*-
# optimizers.py
# author : Antoine Passemiers, Robin Petit

from abc import ABCMeta, abstractmethod
from operator import mul
from functools import reduce
import numpy as np


class Optimizer(metaclass=ABCMeta):
    """ Base class for first order and second order optimizers. """

    def __init__(self):
        self.gradient_fragments = list()
    
    def flush(self):
        self.gradient_fragments = list()
    
    def update(self, grad):
        """ Computes best move in the parameter space at
        current iteration using optimization techniques.
        Input gradient is temporarily flattened as a vector.

        Args:
            grad (np.ndarray): Error gradient with respect to
                the parameters of current layer.
        
        Returns:
            np.ndarray: Array of the same shape as the input,
                representing the best move in parameter space 
                (steepest descent or hessian direction)
                of length determined by the steplength.
        """
        in_shape = grad.shape
        delta = self._update(grad.flatten(order='C'))
        return delta.reshape(in_shape, order='C')
    
    def optimize(self):
        gradient = list()
        for src_layer, layer_param_shapes, fragments in self.gradient_fragments:
            for fragment in fragments:
                gradient.append(fragment.flatten(order='C'))
        gradient = np.concatenate(gradient)

        delta = self._update(gradient)

        cursor = 0
        for src_layer, layer_param_shapes, _ in self.gradient_fragments:
            layer_fragments = list()
            for fragment_shape in layer_param_shapes:
                n_elements = reduce(mul, fragment_shape)
                fragment = delta[cursor:cursor+n_elements]
                layer_fragments.append(fragment.reshape(fragment_shape, order='C'))
                cursor += n_elements
            src_layer.update_parameters(tuple(layer_fragments))

    @abstractmethod
    def _update(self, grad):
        pass
    
    def add_gradient_fragments(self, src_layer, fragments):
        if not (isinstance(fragments, tuple) or (isinstance(fragments, list))):
            fragments = [fragments]
        layer_param_shapes = list()
        for fragment in fragments:
            layer_param_shapes.append(fragment.shape)
        self.gradient_fragments.append((src_layer, layer_param_shapes, fragments))


class MomentumOptimizer(Optimizer):
    """ Simple first order optimizer with momentum support.
    
    Args:
        learning_rate (:obj:`float`, optional): Constant steplength.
        momentum (:obj:`float`, optional): Persistence of previous
            gradient vectors. Old vectors are re-used to compute the
            new search direction, with respect to the momentum value.
    
    Attributes:
        previous_grad (np.ndarray): Gradient vector at previous iteration.
    """

    def __init__(self, learning_rate=.005, momentum=.9):
        Optimizer.__init__(self)
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
    """
    Args:
        learning_rate (:obj:`float`, optional): Constant steplength.
        beta_1 (:obj:`float`, optional): Exponential decay rate of the moving
            average of the gradient.
        beta_2 (:obj:`float`, optional): Exponential decay rate of the moving
            average of th.
        epsilon (:obj:`float`, optional): Constant for numeric stability.
    
    Attributes:
        step (int): Current iteration.
        moment_1 (np.ndarray): Last 1st moment vector.
        moment_2 (np.ndarray): Last 2nd moment vector.

    References:
        ADAM: A Method For Stochastic Optimization
            Diederik P. Kingma and Jimmy Lei Ba
            https://arxiv.org/pdf/1412.6980.pdf
    """

    def __init__(self, learning_rate=.001, beta_1=.9, beta_2=.999, epsilon=1e-8):
        Optimizer.__init__(self)
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
    """ Quasi-newtonian optimizer with limited memory.

    Args:
        m (:obj:`int`, optional): Memory size.
    
    Attributes:
        k (int): Current iteration of L-BFGS.
        previous_grad (np.ndarray): Gradient vector at
            iteration k-1.
        y (list): List of m last gradient differences.
            y_t = grad_{t+1} - grad_t
        s (list): List of m last update vectors.
            s_t = H * grad * steplength, where H is the
            Hessian matrix.
        alpha (list): List of m last alpha coefficients
            alpha_i = rho_i * s_i.T * grad,
            where rho_i = 1. / (s_i.T * y_i).
    
    References:
        Updating Quasi-Newton Matrices with Limited Storage
            Nocedal, J. (1980)
            Mathematics of Computation. 35 (151): 773–782
    """

    def __init__(self, m=10):
        Optimizer.__init__(self)
        self.m = m
        self.k = -1
        self.previous_grad = None
        self.y = list()
        self.s = list()
        self.alpha = list()

    def _update(self, grad):
        # Two-loop recursion: Only if memory contains a sufficient
        # number of update vectors
        if self.k >= self.m:
            q = np.copy(grad)
            for s_i, y_i, alpha_i in reversed(list(zip(self.s, self.y, self.alpha))):
                q -= alpha_i * y_i

            # Implicit product between Hessian matrix and gradient vector
            z = np.repeat(np.dot(self.y[-1] \
                / np.dot(self.y[-1], self.y[-1]), self.s[-1])) * q

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

            # Ensure history has a length of m
            if len(self.y) > self.m:
                self.y = self.y[1:]
                self.s = self.s[1:]
                self.alpha = self.alpha[1:]

        self.previous_grad = grad
        self.k += 1

        return delta
