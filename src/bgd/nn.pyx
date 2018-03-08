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

    def train(self, X, y, steps=10000, error_op='cross-entropy', batch_size=200, learning_rate=.01, alpha=.0001, print_every=50, validation_fraction=0.1):
        batch_size = min(len(X), batch_size)
        error_op = CrossEntropy() if error_op.lower() == 'cross-entropy' else MSE()
        errors = list()

        # Split data into training data and validation data for early stopping
        split = int(len(X) * (1. - validation_fraction))
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], np.squeeze(y[indices])
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        # Binarize labels if classification task
        y_train = self.binarize_labels(y_train)

        # Activate dropout
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.activate()

        epochs, seen_instances = 0, 0
        for step in range(steps):
            for batch_x, batch_y in self.mini_batches(X_train, y_train, batch_size=batch_size):
                # Forward pass
                probs = self.eval(batch_x)
                
                # Compute loss function
                loss = error_op.eval(batch_y, probs)

                # Apply L2 regularization
                for layer in self.layers:
                    params = layer.get_parameters()
                    if params:
                        loss += 0.5 * alpha * np.sum(params[0] ** 2)
                errors.append(loss)

                # Compute gradient of the loss function
                error = error_op.grad(batch_y, probs)

                # Propagate error through each layer
                for layer in reversed(self.layers):
                    extra_info = {'learning_rate': learning_rate, 'l2_reg': alpha}
                    error = layer.backward(error, extra_info)
                
                seen_instances += batch_size
                epochs += 1
                if seen_instances % print_every == 0:
                    if validation_fraction > 0:
                        val_probs = self.eval(X_val)
                        val_accuracy = ((val_probs.argmax(axis=1) == y_val).sum() / len(y_val)) * 100
                    else:
                        val_accuracy = '-'
                    print('Loss at epoch {0}: {1: <10} - Validation accuracy: {2: <10}'.format(
                        str(epochs).ljust(7), loss, val_accuracy))

        # Deactivate dropout
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.deactivate()
        
        return errors
        
    def eval(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X