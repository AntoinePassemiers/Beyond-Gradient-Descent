# -*- coding: utf-8 -*-
# initializers.py
# author : Antoine Passemiers, Robin Petit

from abc import ABCMeta, abstractmethod
import numpy as np


class Initializer(metaclass=ABCMeta):

    def __init__(self, seed=None):
        self.seed = seed
    
    @abstractmethod
    def _initialize(self, shape):
        pass
    
    def initialize(self, shape, dtype=np.float32):
        if self.seed:
            np.random.seed(self.seed)
        return np.asarray(self._initialize(shape), dtype=dtype)


class UniformInitializer(Initializer):

    def __init__(self, min_value, max_value, seed=None):
        Initializer.__init__(self, seed=None)
        self.min_value = min_value
        self.max_value = max_value
    
    def _initialize(self, shape):
        return np.random.uniform(self.min_value, self.max_value, size=shape)


class GaussianInitializer(Initializer):

    def __init__(self, mean, stdv, truncated=False, seed=None):
        Initializer.__init__(self, seed=None)
        self.mean = mean
        self.stdv = stdv
        self.truncated = truncated
    
    def _initialize(self, shape):
        # TODO: if truncated, discard samples that are more than 2*stdv and re-generate them
        return np.random.normal(loc=self.mean, scale=self.stdv, size=shape)


class XavierInitializer(Initializer):

    def __init__(self, seed=None):
        """
        References
        ----------
        http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        """
        Initializer.__init__(self, seed=None)

    def _initialize(self, shape):
        if isinstance(shape, int):
            stdv = 1. / shape
        elif len(shape) == 1:
            stdv = 1. / shape[0]
        elif len(shape) == 2:
            stdv = 2. / (shape[0] + shape[1])
        else:
            raise NotImplementedError()
        return np.random.normal(loc=0, scale=stdv, size=shape)