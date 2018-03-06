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
    
    def mini_batches(self, X, y, batch_size=50):
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]


    def train(self, X, y, steps=10000, error_op='cross-entropy', batch_size=50, learning_rate=.01, reg_L2=.01, print_every=50):
        error_op = CrossEntropy() if error_op.lower() == 'cross-entropy' else MSE()
        errors = list()

        # Activate dropout
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.activate()

        epochs = 0
        for step in range(steps):
            for batch_x, batch_y in self.mini_batches(X, y, batch_size=batch_size):
                # Forward pass
                probs = self.eval(batch_x)
                
                # Compute loss function
                loss = error_op.eval(batch_y, probs)

                # Apply L2 regularization
                for layer in self.layers:
                    params = layer.get_parameters()
                    if params:
                        loss += 0.5 * reg_L2 * np.sum(params[0] ** 2)
                errors.append(loss)

                # Compute gradient of the loss function
                error = error_op.grad(batch_y, probs)

                # Propagate error through each layer
                for layer in reversed(self.layers):
                    extra_info = {'learning_rate': learning_rate, 'l2_reg': reg_L2}
                    error = layer.backward(error, extra_info)
                
                epochs += 1
                if epochs % print_every == 0:
                    print('Loss at epoch {0}: {1}'.format(epochs, loss))

        # Deactivate dropout
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.deactivate()
        
        return errors
        
    def eval(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X