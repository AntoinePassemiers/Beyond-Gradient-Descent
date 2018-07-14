# test.py
# author : Antoine Passemiers, Robin Petit

import numpy as np

from bgd.batch import SGDBatching
from bgd.cost import MSE
from bgd.layers import *
from bgd.nn import NeuralStack
from bgd.optimizers import MomentumOptimizer


def test_mlp_on_xor_problem():
    nn = NeuralStack()
    nn.add(FullyConnected(2, 8))
    nn.add(Activation(function='tanh'))
    nn.add(FullyConnected(8, 1))
    nn.add(Activation(function='sigmoid'))
    nn.add(MSE())
    nn.add(MomentumOptimizer(learning_rate=0.5, momentum=0))
    nn.add(SGDBatching(4))
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    nn.train(X, y, l2_alpha=0, epochs=1000, print_every=800)
    predictions = np.squeeze(nn.eval(X))
    assert np.array_equal(predictions > 0.5, y)

def test_learnable():
    non_learnable_layers = list()
    non_learnable_layers.append(Activation(function='tanh'))
    non_learnable_layers.append(Flatten())
    non_learnable_layers.append(Dropout())
    non_learnable_layers.append(GaussianNoise(0, 1))
    non_learnable_layers.append(MaxPooling2D((2, 2)))
    non_learnable_layers.append(Lambda(lambda x: 2*x, lambda x: 0.5*x))
    for layer in non_learnable_layers:
        assert not layer.learnable()
    learnable_layers = list()
    learnable_layers.append(Convolutional2D((3, 3, 4), 16))
    learnable_layers.append(FullyConnected(50, 70))
    for layer in learnable_layers:
        assert layer.learnable()

