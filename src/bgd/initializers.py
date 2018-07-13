""" This module contains contains all the initializers that
are implemented. Any new initializer needs to inherit from
:class:`bgd.initializers.Initializer` and to implement its
abstract methods (:obj:`_initialize`). """

# initializers.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'ZeroInitializer', 'UniformInitializer', 'GlorotUniformInitializer',
    'GaussianInitializer', 'GlorotGaussianInitializer'
]

from abc import ABCMeta, abstractmethod
import numpy as np

# pylint: disable=too-few-public-methods

class Initializer(metaclass=ABCMeta):
    """ Base class for all initializers.

    Args:
        seed (int):
            Seed for the random number generator.
    """

    def __init__(self, seed=None):
        self.seed = seed

    def initialize(self, shape, dtype=np.float32):
        """ Return array with random values. The distribution
        of the values is defined in subclasses.

        Args:
            shape (tuple):
                Shape of the array to be initialized.
            dtype (np.dtype):
                Data type of the array to be initialized.

        Returns:
            :obj:`np.ndarray`:
                An array of provided shape initialized accordingly.
        """
        if self.seed:
            np.random.seed(self.seed)
        return np.asarray(self._initialize(shape), dtype=dtype)

    @abstractmethod
    def _initialize(self, shape):
        pass


class ZeroInitializer(Initializer):
    """ Initializer for generating arrays of zeroes.

    Args:
        seed (int):
            Seed for the random number generator.
    """

    def __init__(self):
        Initializer.__init__(self, seed=None)

    def _initialize(self, shape):
        return np.zeros(shape)

class UniformInitializer(Initializer):
    """ Initializer for generating arrays using a uniform distribution.

    Args:
        min_value (float):
            Lower bound of the uniform distribution.
        max_value (float):
            Upper bound of the uniform distribution.
        seed (int):
            Seed for the random number generator.
    """

    def __init__(self, min_value=-.05, max_value=.05, seed=None):
        Initializer.__init__(self, seed=seed)
        self.min_value = min_value
        self.max_value = max_value

    def _initialize(self, shape):
        return np.random.uniform(self.min_value, self.max_value, size=shape)


class GlorotUniformInitializer(Initializer):
    """ Initializer for generating arrays using a
    Glorot uniform distribution.

    Args:
        seed (int):
            Seed for the random number generator.
    """

    def __init__(self, seed=None):
        Initializer.__init__(self, seed=seed)

    def _initialize(self, shape):
        if isinstance(shape, int):
            limit = np.sqrt(6. / shape)
        elif len(shape) == 1:
            limit = np.sqrt(1. / shape[0])
        elif len(shape) == 2:
            limit = np.sqrt(2. / (shape[0] + shape[1]))
        else:
            limit = np.sqrt(len(shape) / np.sum(shape))
        return np.random.uniform(-limit, limit, size=shape)


class GaussianInitializer(Initializer):
    """ Initializer for generating arrays using a
    Gaussian distribution.

    Args:
        mean (float):
            Mean of the Gaussian distribution.
        stdv (float):
            Standard deviation of the Gaussian distribution.
        truncated (bool):
            Whether to truncate the sampled values.
        seed (int):
            Seed for the random number generator.
    """

    def __init__(self, mean, stdv, truncated=False, seed=None):
        Initializer.__init__(self, seed=seed)
        self.mean = mean
        self.stdv = stdv
        self.truncated = truncated

    def _initialize(self, shape):
        ret = np.random.normal(loc=self.mean, scale=self.stdv, size=shape)
        loop = self.truncated
        while loop:
            centered_ret = ret - self.mean
            bound = 2*self.stdv
            is_out_of_bounds = np.logical_or(centered_ret < bound, -bound < centered_ret)
            if is_out_of_bounds.any():
                indices = np.where(is_out_of_bounds)
                size = len(indices[0]) if isinstance(indices, tuple) else len(indices)
                ret[indices] = np.random.normal(loc=self.mean, scale=self.stdv, size=size)
            else:
                loop = False
        return ret


class GlorotGaussianInitializer(Initializer):
    """ Initializer for generating arrays using a
    Glorot Gaussian distribution.

    Args:
        seed (int):
            Seed for the random number generator
    """

    def __init__(self, seed=None):
        Initializer.__init__(self, seed=seed)

    def _initialize(self, shape):
        if isinstance(shape, int):
            stdv = np.sqrt(2. / shape)
        elif len(shape) == 1:
            stdv = np.sqrt(2. / shape[0])
        elif len(shape) == 2:
            stdv = np.sqrt(2. / (shape[0] + shape[1]))
        else:
            raise NotImplementedError()
        return np.random.normal(loc=0, scale=stdv, size=shape)
