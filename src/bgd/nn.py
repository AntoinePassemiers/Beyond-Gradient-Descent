# -*- coding: utf-8 -*-
# nn.py
# author : Antoine Passemiers, Robin Petit

from bgd.batch import Batching, SGDBatching
from bgd.errors import Error
from bgd.layers import Layer, Dropout
from bgd.errors import MSE, CrossEntropy
from bgd.optimizers import MomentumOptimizer, Optimizer
from bgd.utils import log, RequiredComponentError, WrongComponentTypeError

import copy
import numpy as np


class NeuralStack:

    def __init__(self):
        self.layers = list()
        self.batch_op = None
        self.error_op = None
        self.optimizer = None

    def add(self, component):
        if isinstance(component, Layer):
            self.layers.append(component)
        elif isinstance(component, Optimizer):
            self.optimizer = component
        elif isinstance(component, Error):
            self.error_op = component
        elif isinstance(component, Batching):
            self.batch_op = component
        else:
            raise WrongComponentTypeError("Unknown component type")

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
    
    def eval_loss(self, batch_y, predictions, alpha):
        loss = self.error_op.eval(batch_y, predictions)

        # Apply L2 regularization
        if alpha > 0:
            for layer in self.layers:
                params = layer.get_parameters()
                if params:
                    squared_params = params[0] ** 2
                    if not np.isnan(squared_params).any():
                        loss += 0.5 * alpha * np.sum(squared_params)
        return loss

    def train(self, X, y, epochs=1000, batch_size=200, alpha_reg=.0001,
              print_every=50, validation_fraction=0.1):

        errors = list()

        # Split data into training data and validation data for early stopping
        if validation_fraction > 0:
            X_train, y_train, X_val, y_val = self.split_train_val(X, y, validation_fraction)
        else:
            X_train, y_train = X, y

        # Binarize labels if classification task
        if isinstance(self.error_op, CrossEntropy):
            y_train = self.binarize_labels(y_train)

        # Deactivate signal propagation though first layer
        self.layers[0].deactivate_propagation()

        seen_instances = 0
        for epoch in range(epochs):
            batch_id = 0
            self.batch_op.start(X_train, y_train)
            next_batch = self.batch_op.next()
            while next_batch:
                # Retrieve batch for next iteration
                (batch_x, batch_y) = next_batch
                next_batch = self.batch_op.next()

                # Forward pass
                predictions = self.eval(batch_x)

                # Compute loss function
                alpha = alpha_reg / self.batch_op.batch_size
                loss = self.eval_loss(batch_y, predictions, alpha_reg)
                errors.append(loss)

                # Compute gradient of the loss function
                signal = self.error_op.grad(batch_y, predictions)

                # Propagate error through each layer
                for layer in reversed(self.layers):
                    extra_info = {'l2_reg': alpha}
                    signal, gradient = layer.backward(signal, extra_info)
                    if gradient is not None:
                        self.optimizer.add_gradient_fragments(layer, gradient)
                F = lambda: self.eval_loss(batch_y, self.eval(batch_x), alpha)
                self.optimizer.update(F)
                self.optimizer.flush()

                seen_instances += self.batch_op.batch_size
                if seen_instances % print_every == 0:
                    batch_id += 1
                    # Warning: This code section is ugly
                    if isinstance(self.error_op, CrossEntropy):
                        if validation_fraction > 0:
                            val_accuracy = self.get_accuracy(X_val, y_val)
                        else:
                            val_accuracy = -1
                        log('Loss at epoch {0} (batch {1: <9} : {2: <20} - Validation accuracy: {3:.1f}'.format(
                            epoch, str(batch_id) + ')', loss, val_accuracy))
                    else:
                        if validation_fraction > 0:
                            val_preds = self.eval(X_val)
                            val_mse = self.error_op.eval(y_val, val_preds)
                        else:
                            val_accuracy = -1
                        log('Loss at epoch {0} (batch {1: <9} : {2: <20} - Validation MSE: {3: <15}'.format(
                            epoch, str(batch_id) + ')', loss, val_mse))
        return errors

    def eval(self, X, start=0, stop=-1):
        if stop == -1 or stop > len(self.layers):
            stop = len(self.layers) 
        for i in range(start, stop):
            X = self.layers[i].forward(X)
        return X

    def activate_dropout(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.activate()
    
    def deactivate_dropout(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.deactivate()

    def check_components(self):
        # Check batch optimization method
        if self.batch_op is None:
            raise RequiredComponentError(Batching.__name__)

        # Check optimizer
        if self.optimizer is None:
            raise RequiredComponentError(Optimizer.__name__)

        # Check loss function
        if self.error_op is None:
            raise RequiredComponentError(Error.__name__)