# -*- coding: utf-8 -*-
# nn.pyx
# author : Antoine Passemiers, Robin Petit
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

from bgd.layers import Dropout
from bgd.errors import MSE, CrossEntropy
from bgd.optimizers import MomentumOptimizer, Optimizer
from bgd.utils import log

import copy

import numpy as np


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
            yield X[i:i + batch_size], y[i:i + batch_size]

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

    def get_accuracy(self, X_val, y_val, batch_size=256):
        nb_correct = 0
        for i in np.arange(0, X_val.shape[0], batch_size):
            indices = i + np.arange(min(batch_size, X_val.shape[0]-i))
            val_probs = self.eval(X_val[indices])
            assert len(val_probs) == len(indices)
            nb_correct += (val_probs.argmax(axis=1) == y_val[indices]).sum()
        return 100 * nb_correct / X_val.shape[0]

    def train(self, X, y, epochs=1000, optimizer='default', error_op='cross-entropy',
              batch_size=200, alpha=.0001, print_every=50, validation_fraction=0.1):
        batch_size = min(len(X), batch_size)
        alpha /= batch_size
        error_op = CrossEntropy() if error_op.lower() == 'cross-entropy' else MSE()
        errors = list()

        # Check optimizer
        if optimizer == 'default':
            optimizer = MomentumOptimizer(learning_rate=.001, momentum=.9)
        assert(isinstance(optimizer, Optimizer))

        # Split data into training data and validation data for early stopping
        if validation_fraction > 0:
            X_train, y_train, X_val, y_val = self.split_train_val(X, y, validation_fraction)
        else:
            X_train, y_train = X, y

        # Binarize labels if classification task
        if isinstance(error_op, CrossEntropy):
            y_train = self.binarize_labels(y_train)

        # Activate dropout
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.activate()

        # Create one independent optimizer per layer
        for layer in self.layers:
            layer.set_optimizer(copy.deepcopy(optimizer))

        seen_instances = 0
        for epoch in range(epochs):
            for batch_id, (batch_x, batch_y) in enumerate(self.mini_batches(X_train, y_train, batch_size=batch_size)):
                # Forward pass
                predictions = self.eval(batch_x)

                # Compute loss function
                loss = error_op.eval(batch_y, predictions)

                # Apply L2 regularization
                if alpha > 0:
                    for layer in self.layers:
                        params = layer.get_parameters()
                        if params:
                            loss += 0.5 * alpha * np.sum(params[0] ** 2)
                errors.append(loss)

                # Compute gradient of the loss function
                error = error_op.grad(batch_y, predictions)

                # Propagate error through each layer
                for layer in reversed(self.layers):
                    extra_info = {'l2_reg': alpha}
                    error = layer.backward(error, extra_info)

                seen_instances += batch_size
                if seen_instances % print_every == 0:
                    batch_id += 1
                    # Warning: This code section is ugly
                    if isinstance(error_op, CrossEntropy):
                        if validation_fraction > 0:
                            val_accuracy = self.get_accuracy(X_val, y_val)
                        else:
                            val_accuracy = -1
                        log('Loss at epoch {0} (batch {1: <9} : {2: <20} - Validation accuracy: {3:.1f}'.format(
                            epoch, str(batch_id) + ')', loss, val_accuracy))
                    else:
                        if validation_fraction > 0:
                            val_preds = self.eval(X_val)
                            val_mse = error_op.eval(batch_y, val_preds)
                        else:
                            val_accuracy = -1
                        log('Loss at epoch {0} (batch {1: <9} : {2: <20} - Validation MSE: {3: <15}'.format(
                            epoch, str(batch_id) + ')', loss, val_mse))

        # Deactivate dropout
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.deactivate()

        return errors

    def eval(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
