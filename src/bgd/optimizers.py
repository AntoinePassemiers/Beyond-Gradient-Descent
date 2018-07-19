""" This module contains all the optimizers that are implemented.
Any optimizer implemented needs to inherit from
:class:`bgd.optimizers.Optimizer` and to implement its abstract
method (:obj:`update`). """

# optimizers.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'MomentumOptimizer', 'AdamOptimizer', 'LBFGS'
]

from abc import ABCMeta, abstractmethod
from operator import mul
from functools import reduce
import numpy as np


class Optimizer(metaclass=ABCMeta):
    """ Base class for first order and second order optimizers.

    Attributes:
        gradient_fragments (list): List of tuples of NumPy arrays,
            where the number of tuples is equal to the number of
            learnable layers in the network.
    """

    def __init__(self):
        self.gradient_fragments = list()

    def flush(self):
        self.gradient_fragments = list()

    @abstractmethod
    def _update(self, grad, F):
        pass

    def update(self, F):
        """ Computes best move in the parameter space at
        current iteration using optimization techniques.
        All gradient fragments added to gradient_fragments
        are flattened and concatenated to get the batch
        gradient vector of the whole network.
        The optimized delta vector is then split into
        several fragments of original shapes. Finally,
        those delta fragments are used to update the parameters
        of each layer, individually.
        """
        gradient = list()
        for _, _, fragments in self.gradient_fragments:
            for fragment in fragments:
                gradient.append(fragment.flatten(order='C'))
        gradient = np.concatenate(gradient)

        delta = self._update(gradient, F)
        self.update_layers(delta)
        self.flush()

    def update_layers(self, delta):
        cursor = 0
        for src_layer, layer_param_shapes, _ in self.gradient_fragments:
            layer_fragments = list()
            for fragment_shape in layer_param_shapes:
                n_elements = reduce(mul, fragment_shape)
                fragment = delta[cursor:cursor+n_elements]
                layer_fragments.append(fragment.reshape(fragment_shape, order='C'))
                cursor += n_elements
            src_layer.update_parameters(tuple(layer_fragments))

    def add_gradient_fragments(self, src_layer, fragments, l2_alpha=0.):
        if isinstance(fragments, tuple):
            fragments = list(fragments)
        elif not isinstance(fragments, list):
            fragments = [fragments]
        layer_param_shapes = list(map(lambda f: f.shape, fragments))
        # L2 regularization
        if l2_alpha > 0:
            layer_params = src_layer.get_parameters()
            for i in range(len(fragments)):
                assert fragments[i].shape == layer_params[i].shape
                fragments[i] += l2_alpha * layer_params[i]
        self.gradient_fragments.append((src_layer, layer_param_shapes, fragments))


class MomentumOptimizer(Optimizer):
    """ Simple first order optimizer with momentum support.

    Args:
        learning_rate (:obj:`float`, optional):
            Constant steplength.
        momentum (:obj:`float`, optional):
            Persistence of previous gradient vectors. Old vectors
            are re-used to compute the new search direction,
            with respect to the momentum value.

    Attributes:
        previous_grad (:obj:`np.ndarray`):
            Gradient vector at previous iteration.
    """

    def __init__(self, learning_rate=.005, momentum=.9):
        Optimizer.__init__(self)
        assert 0 <= momentum <= 1
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.previous_grad = None

    def _update(self, grad, F):
        delta = self.learning_rate * grad
        if self.momentum > 0:
            if self.previous_grad is not None:
                delta += self.momentum * self.previous_grad
            self.previous_grad = delta
        return delta


class AdamOptimizer(Optimizer):
    """
    Args:
        learning_rate (:obj:`float`, optional):
            Constant steplength.
        beta_1 (:obj:`float`, optional):
            Exponential decay rate of the moving average of the gradient.
        beta_2 (:obj:`float`, optional):
            Exponential decay rate of the moving average of th.
        epsilon (:obj:`float`, optional):
            Constant for numeric stability.

    Attributes:
        step (int): Current iteration.
        moment_1 (:obj:`np.ndarray`):
            Last 1st moment vector.
        moment_2 (:obj:`np.ndarray`):
            Last 2nd moment vector.

    References:
        ADAM: A Method For Stochastic Optimization
            Diederik P. Kingma and Jimmy Lei Ba
            https://arxiv.org/pdf/1412.6980.pdf
    """

    def __init__(self, learning_rate=.001, beta_1=.9, beta_2=.999, epsilon=1e-8):
        Optimizer.__init__(self)
        assert (0 <= beta_1 < 1) and (0 <= beta_2 < 1)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.step = 0
        self.moment_1 = 0
        self.moment_2 = 0

    def _update(self, grad, F):
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
        m (:obj:`int`, optional):
            Memory size.
        epsilon (:obj:`float`, optional):
            Constant for numeric stability.
        first_order_optimizer (:class:`Optimizer`):
            First order optimizer used to approximate the Hessian product.

    Attributes:
        k (int):
            Current iteration of L-BFGS.
        previous_grad (:obj:`np.ndarray`):
            Gradient vector at iteration k-1.
        y (list):
            List of m last gradient differences.
            y_t = grad_{t+1} - grad_t
        s (list):
            List of m last update vectors.
            s_t = H * grad * steplength, where H is the
            Hessian matrix.
        alpha (list):
            List of m last alpha coefficients
            alpha_i = rho_i * s_i.T * grad,
            where rho_i = 1. / (s_i.T * y_i).

    References:
        Updating Quasi-Newton Matrices with Limited Storage
            Nocedal, J. (1980)
            Mathematics of Computation. 35 (151): 773–782
    """

    def __init__(self, m=10, epsilon=1e-02, first_order_optimizer=AdamOptimizer()):
        Optimizer.__init__(self)
        self.first_order_optimizer = first_order_optimizer
        self.epsilon = epsilon
        self.m = m
        self.k = -1
        self.previous_grad = None
        self.y = list()
        self.s = list()

    def _update(self, grad, F):
        if self.previous_grad is not None:
            y_k_minus_1 = grad - self.previous_grad
            s_k_minus_1 = self.s[-1]
            if np.dot(y_k_minus_1, s_k_minus_1) > self.epsilon * np.sum(s_k_minus_1 ** 2) or True: # TODO
                # Quasi-Newton update
                self.y.append(y_k_minus_1)
                ## TODO: check for rho_k_minus_1 because it is not used
                #rho_k_minus_1 = 1. / np.dot(s_k_minus_1, y_k_minus_1)
            else:
                self.s = self.s[:-1]
            # Ensure history has a length of m
            if len(self.y) > self.m:
                self.y = self.y[1:]
                self.s = self.s[1:]
            # Ensure that correction pairs are actually pairs
            assert len(self.s) == len(self.y)


        # Two-loop recursion: Only if memory contains a sufficient
        # number of update vectors
        if self.k >= self.m and False:
            q = np.copy(grad)
            alpha = list()
            for s_i, y_i in reversed(list(zip(self.s, self.y))):
                rho_i = 1. / np.dot(s_i, y_i)
                alpha_i = rho_i * np.dot(s_i, q)
                q -= alpha_i * y_i
                alpha.append(alpha_i)

            # Implicit product between Hessian matrix and gradient vector
            den = np.dot(self.y[-1], self.y[-1])
            if den > 0:
                z = (np.dot(self.y[-1], self.s[-1]) / den) * q
            else:
                z = np.zeros(len(q))

            for s_i, y_i, alpha_i in zip(self.s, self.y, list(reversed(alpha))):
                rho_i = 1. / np.dot(s_i, y_i)
                beta_i = rho_i * np.dot(y_i, z)
                z += s_i * (alpha_i - beta_i)
            # At this point z is now an approximation of np.dot(H_k, grad)
            # The search direction is p_k = -z

            # Line search
            c1 = 1e-04
            steplength = 1.
            f_value = F()
            armijo_cnd_satisfied = False
            last_delta = 0
            while (not armijo_cnd_satisfied) and (steplength > 1e-15):
                delta = steplength * z
                self.update_layers(delta - last_delta)
                f_prime_value = F()
                last_delta = delta
                armijo_cnd_satisfied = (f_prime_value <= f_value \
                    - c1*steplength*np.dot(grad, z))
                if not armijo_cnd_satisfied:
                    steplength /= 2.
            # print(steplength, np.mean(np.abs(z)))
            self.update_layers(-delta)
        else:
            # Gradient mode
            delta = self.first_order_optimizer._update(grad, F)

        self.s.append(-delta)
        self.previous_grad = grad
        self.k += 1

        return delta
