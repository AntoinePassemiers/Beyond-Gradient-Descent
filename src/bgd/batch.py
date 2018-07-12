""" This module contains all the different batching methods
that are implemented. Any new batching method needs to inherit
from :class:`bgd.batch.Batching` and to implement its abstract
methods (:obj:`_start` and :obj:`next`). """

# batch.py: Batching methods for neural networks
# author : Antoine Passemiers, Robin Petit

__all__ = ['SGDBatching']

from abc import ABCMeta, abstractmethod
import numpy as np

class Batching(metaclass=ABCMeta):
    """ Base class for generating batches.

    Attributes:
        warm (bool):
            Whether a dataset has been passed to the
            batching algorithm. Once it is warm,
            method 'next' can be called to retrieve
            a batch.
    """

    def __init__(self):
        self.warm = False

    def start(self, X, y):
        """ Wrapper method for providing a dataset
        to the batching algorithm.
        This *must* be called before to method :meth:`next` is called.

        Args:
            X (:obj:`np.ndarray`): Input samples
            y (:obj:`np.ndarray`): Target values
        """
        self._start(X, y)
        self.warm = True

    def next(self):
        """ Wrapper method for retrieving a batch from
        the provided dataset.

        Returns:
            :
                X (:obj:`np.ndarray`):
                    Batch samples.
                y (:obj:`np.ndarray`):
                    Batch target values.
        """
        assert self.warm
        return self._next()

    @abstractmethod
    def _start(self, X, y):
        """ Wrapped method for providing a dataset
        to the batching algorithm. Subclasses must
        override this method."""
        pass

    @abstractmethod
    def _next(self):
        """ Wrapped method for retrieving a batch from
        the provided dataset. Subclasses must override
        this method."""
        pass


class SGDBatching(Batching):
    """ Stochastic Gradient Descent Batching algorithm.

    Args:
        batch_size (int):
            Size of each random batch.
        shuffle (bool):
            Whether to shuffle the dataset before
            extracting a batch from it.
    """

    def __init__(self, batch_size, shuffle=True):
        Batching.__init__(self)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches = None

    def _start(self, X, y):
        """ Provide a dataset to the batching algorithm.
        This *must* be called before to method :meth:`next` is called.

        Args:
            X (:obj:`np.ndarray`): Input samples.
            y (:obj:`np.ndarray`): Target values.
        """
        self.batch_size = min(len(X), self.batch_size)
        self.batches = self.mini_batches(X, y)

    def _next(self):
        """ Retrieves next batch using a generator function.

        Returns:
            :
                X (:obj:`np.ndarray`):
                    Batch samples.
                y (:obj:`np.ndarray`):
                    Batch target values.
        """
        return next(self.batches)

    def mini_batches(self, X, y):
        """ Generator function that iteratively yields batches.

        Args:
            X (:obj:`np.ndarray`): Input samples.
            y (:obj:`np.ndarray`): Target values.

        Yields:
            :
                X (:obj:`np.ndarray`):
                    Batch samples.
                y (:obj:`np.ndarray`):
                    Batch target values.
        """
        indices = np.arange(0, len(X), self.batch_size)
        if self.shuffle: # If shuffled dataset
            np.random.shuffle(indices)
        for i in indices:
            yield X[i:i + self.batch_size], y[i:i + self.batch_size]
        yield None
