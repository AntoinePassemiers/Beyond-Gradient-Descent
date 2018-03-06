# -*- coding: utf-8 -*-
# nn.pyx
# author : Antoine Passemiers, Robin Petit
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

from bgd.layers import Activation, FullyConnected, Dropout
from bgd.errors import MSE, CrossEntropy

import numpy as np
cimport numpy as cnp
cnp.import_array()


class NeuralStack:

    def __init__(self):
        self.layers = list()
        
    def add(self, layer):
        self.layers.append(layer)

    def train(self, X, y, steps=10000, error_op='cross-entropy', learning_rate=.01, reg_L2=.01, print_every=50):
        error_op = CrossEntropy() if error_op.lower() == 'cross-entropy' else MSE()
        errors = list()

        # Activate dropout
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.activate()

        for step in range(steps):
            # Forward pass
            probs = self.eval(X)
            
            # Compute loss function
            loss = error_op.eval(y, probs)

            # Apply L2 regularization
            for layer in self.layers:
                params = layer.get_parameters()
                if params:
                    loss += 0.5 * reg_L2 * np.sum(params[0] ** 2)
            errors.append(loss)

            # Compute gradient of the loss function
            error = error_op.grad(y, probs)

            # Propagate error through each layer
            for layer in reversed(self.layers):
                extra_info = {'learning_rate': learning_rate, 'l2_reg': reg_L2}
                error = layer.backward(error, extra_info)
            
            if step % print_every == 0:
                print('Loss at step {0}: {1}'.format(step, loss))

        # Deactivate dropout
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.deactivate()
        
        return errors
        
    def eval(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X