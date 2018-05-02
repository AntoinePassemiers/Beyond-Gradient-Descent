# -*- coding: utf-8 -*-
# batch.py: Batching methods for neural networks
# author : Antoine Passemiers, Robin Petit

import numpy as np
from abc import ABCMeta, abstractmethod


class Batching(metaclass=ABCMeta):

    def __init__(self):
        self.warm = False
    
    def start(self, X, y):
        self._start(X, y)
        self.warm = True
    
    @abstractmethod
    def _start(self, X, y):
        pass
    
    def next(self):
        assert(self.warm)
        return self._next()

    @abstractmethod
    def _next(self):
        pass


class SGDBatching(Batching):

    def __init__(self, batch_size, shuffle=True):
        Batching.__init__(self)
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def _start(self, X, y):
        self.batch_size = min(len(X), self.batch_size)
        self.batches = self.mini_batches(X, y)
    
    def _next(self):
        return next(self.batches)

    def mini_batches(self, X, y):
        indices = np.arange(0, len(X), self.batch_size)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield X[i:i + self.batch_size], y[i:i + self.batch_size]
        yield None