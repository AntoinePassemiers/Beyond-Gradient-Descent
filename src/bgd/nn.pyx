# -*- coding: utf-8 -*-
# nn.pyx
# author : Antoine Passemiers, Robin Petit
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
#cimport numpy as np

LEARNING_RATE = .1

def sigmoid(z, derivative=False):
    if derivative:
        return z*(1-z)
    return 1. / (1 + np.exp(-z))

def mse(x, y, derivative=False):
    if derivative:
        return y-x
    return .5*(y-x)**2

def init_weights(shape):
    return np.random.random(shape) - 1

class NeuralNet:
    def __init__(self):
        self.layers = list()
        
    def add_layer(self, shape, activator=sigmoid):
        self.layers.append(FullyConnected(shape, activator))
        
    def add_layers(self, shape, activators=sigmoid):
        for i in range(len(shape)-1):
            activator = activators[i] if isinstance(activators, tuple) else activators
            self.layers.append(FullyConnected((shape[i], shape[i+1]), activator))
        
    def eval(self, input):
        for layer in self.layers:
            input = layer.feed(input)
        return input
    
    def train_gd(self, X, y, steps: int=10000, error_fct=mse):
        errors = list()
        for step in range(steps):
            if step % 50 == 0:
                print('Step {}'.format(step))
            inputs = [X]
            in_ = X
            for layer in self.layers:
                in_ = layer.feed(in_)
                inputs.append(in_)
            error = error_fct(in_, y, derivative=True)
            errors.append(np.mean(np.abs(error)))
            for layer_id, layer in reversed(list(enumerate(self.layers))):
                delta = error * layer.activator(inputs[layer_id+1], derivative=True)
                layer.update(np.dot(inputs[layer_id].T, delta))
                error = np.dot(delta, layer.weights.T)
        return errors

class FullyConnected:
    def __init__(self, shape, activator, init_fct=init_weights):
        self.weights = init_fct(shape)
        self.activator = activator
    
    def feed(self, in_):
        return self.activator(np.dot(in_, self.weights))
    
    def update(self, error):
        self.weights += error * LEARNING_RATE
