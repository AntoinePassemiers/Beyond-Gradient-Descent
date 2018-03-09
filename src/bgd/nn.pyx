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
from bgd.optimizers import MomentumOptimizer, Optimizer

import copy

import numpy as np
cimport numpy as cnp
cnp.import_array()


class NeuralStack:

    def __init__(self):
        self.layers = list()
        
    def add(self, layer):
        self.layers.append(layer)
    
    def mini_batches(self, X, y, batch_size=50, shuffle=True):
        indices = np.arange(0, len(X), batch_size)
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield X[i:i+batch_size], y[i:i+batch_size]

    def binarize_labels(self, y):
        unique_values = np.unique(y)
        n_classes = len(unique_values)
        binary_y = np.zeros((len(y), n_classes), dtype=np.int)
        for c in range(n_classes):
            binary_y[:, c] = (y == c)
        return binary_y
    
    def split_train_val(self, X, y, validation_fraction):
        split = int(len(X) * (1. - validation_fraction))
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], np.squeeze(y[indices])
        return X[:split], y[:split], X[split:], y[split:]

    def train(self, X, y, epochs=10000, optimizer='default', error_op='cross-entropy', batch_size=200, alpha=.0001, print_every=50, validation_fraction=0.1):
        batch_size = min(len(X), batch_size)
        error_op = CrossEntropy() if error_op.lower() == 'cross-entropy' else MSE()
        errors = list()

        # Check optimizer
        if optimizer == 'default':
            optimizer = MomentumOptimizer(learning_rate=.001, momentum=.9)
        assert(isinstance(optimizer, Optimizer))

        # Split data into training data and validation data for early stopping
        X_train, y_train, X_val, y_val = self.split_train_val(X, y, validation_fraction)

        # Binarize labels if classification task
        y_train = self.binarize_labels(y_train)

        # Activate dropout
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.activate()

        # Create one independent optimizer per layer
        optimizers = list()
        for layer in self.layers:
            optimizers.append(copy.deepcopy(optimizer))

        seen_instances = 0
        for epoch in range(epochs):
            for batch_id, (batch_x, batch_y) in enumerate(self.mini_batches(X_train, y_train, batch_size=batch_size)):
                # Forward pass
                probs = self.eval(batch_x)
                
                # Compute loss function
                loss = error_op.eval(batch_y, probs)

                # Apply L2 regularization
                if alpha > 0:
                    for layer in self.layers:
                        params = layer.get_parameters()
                        if params:
                            loss += 0.5 * alpha * np.sum(params[0] ** 2)
                errors.append(loss)

                # Compute gradient of the loss function
                error = error_op.grad(batch_y, probs)

                # Propagate error through each layer
                for layer, optimizer in zip(reversed(self.layers), reversed(optimizers)):
                    error = np.copy(error)
                    extra_info = {'optimizer': optimizer, 'l2_reg': alpha}
                    error = layer.backward(error, extra_info)
                
                seen_instances += batch_size
                if seen_instances % print_every == 0:
                    if validation_fraction > 0:
                        val_probs = self.eval(X_val)
                        val_accuracy = ((val_probs.argmax(axis=1) == y_val).sum() / len(y_val)) * 100
                    else:
                        val_accuracy = '-'
                    print('Loss at epoch {0} (batch {1: <9} : {2: <20} - Validation accuracy: {3: <15}'.format(
                        epoch, str(batch_id) + ')', loss, val_accuracy))

        # Deactivate dropout
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.deactivate()
        
        return errors
        
    def eval(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X