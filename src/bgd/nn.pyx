# -*- coding: utf-8 -*-
# nn.pyx
# author : Antoine Passemiers, Robin Petit
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

from bgd.layers import Activation, FullyConnected
from bgd.errors import MSE, CrossEntropy

import numpy as np
cimport numpy as cnp
cnp.import_array()


class NeuralStack:

    def __init__(self):
        self.layers = list()
        
    def add(self, layer):
        self.layers.append(layer)

    def train(self, X, y, steps=10000, error_op='cross-entropy', learning_rate=.01, reg_L2=.01):
        error_op = CrossEntropy() if error_op.lower() == 'cross-entropy' else MSE()
        errors = list()

        all_weights = list()
        for layer in self.layers:
            params = layer.get_parameters()
            if params:
                all_weights.append(params[0])

        for step in range(steps):
            probs = self.eval(X)
            
            loss = error_op.eval(y, probs)
            for weights in all_weights:
                loss += 0.5 * reg_L2 * np.sum(weights ** 2)
            errors.append(loss)

            # compute error
            error = error_op.grad(y, probs)
            
            for i in [1, 0]:
                full_layer = self.layers[i*2]
                W, b = full_layer.get_parameters()
                current_input = self.layers[i*2].current_input
                activation_layer = self.layers[i*2+1]
                error = activation_layer._backward(error)
            
                gradient_weights = np.dot(current_input.T, error)
                gradient_weights += reg_L2 * W
                gradient_bias = np.sum(error, axis=0, keepdims=True)
                W -= learning_rate * gradient_weights
                b -= learning_rate * gradient_bias
                error = np.dot(error, W.T)
            
            if step % 50 == 0:
                print('Loss at step {0}: {1}'.format(step, loss))
        
        return errors
        
    def eval(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X