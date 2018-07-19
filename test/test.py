# test.py
# author : Antoine Passemiers, Robin Petit

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from bgd.batch import SGDBatching
from bgd.cost import MSE, CrossEntropy, ClassificationCost
from bgd.layers import *
from bgd.nn import NeuralStack
from bgd.optimizers import MomentumOptimizer, AdamOptimizer


def test_mlp_on_xor_problem():
    nn = NeuralStack()
    nn.add(FullyConnected(2, 8))
    nn.add(Activation(function='tanh'))
    nn.add(FullyConnected(8, 1))
    nn.add(Activation(function='sigmoid'))
    nn.add(MSE())
    nn.add(MomentumOptimizer(learning_rate=.5, momentum=0))
    nn.add(SGDBatching(4))
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    nn.train(X, y, l2_alpha=0, epochs=1000, print_every=10)
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

def test_mlp_digits():
    images, labels = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size=.9, test_size=.1)
    nn = NeuralStack()
    nn.add(FullyConnected(64, 32))
    nn.add(Activation('relu'))
    nn.add(FullyConnected(32, 10))
    nn.add(Activation('softmax'))
    nn.add(SGDBatching(len(X_train)))
    nn.add(CrossEntropy())
    nn.add(AdamOptimizer(learning_rate=5e-3))
    nn.train(X_train, y_train, print_every=1, epochs=100, l2_alpha=.05)
    assert ClassificationCost.accuracy(y_test, nn.eval(X_test)) >= .9

def test_bounds_activations():
    bounds = {
        'softmax': (0, 1),
        'sigmoid': (0, 1),
        'tanh': (-1, +1)
    }
    X = np.random.normal(loc=0, scale=1, size=(1000, 1000))
    for activation in bounds:
        activation_layer = Activation(activation)
        output = activation_layer.forward(X)
        m, M = bounds[activation]
        assert not np.logical_or(output < m, output > M).any()

def test_relu():
    X = np.array(
        [[-1, 5, 2, 4, -1],
         [2, -2, 3, -1, 0],
         [-1, -2, -3, -4, 0]
        ]
    )
    layer = Activation('relu')
    expected_output_forward = np.array(
        [[0, 5, 2, 4, 0],
         [2, 0, 3, 0, 0],
         [0, 0, 0, 0, 0]
        ]
    )
    assert (layer.forward(X) == expected_output_forward).all()
    error = np.ones((3, 5))
    expected_output_backward = np.array(
        [[0, 1, 1, 1, 0],
         [1, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]
        ]
    )
    assert (layer.backward(error)[0] == expected_output_backward).all()

def test_cnn_digits():
    images, labels = load_digits(return_X_y=True)
    images = images.reshape((-1, 8, 8))
    X_train, X_test, y_train, y_test = train_test_split(images, labels, train_size=.9, test_size=.1)
    nn = NeuralStack()
    nn.add(Convolutional2D((4, 4, 1), 16))
    nn.add(Activation('relu'))
    nn.add(MaxPooling2D((2, 2)))
    nn.add(Flatten())
    nn.add(FullyConnected(256, 64))
    nn.add(Activation('sigmoid'))
    nn.add(FullyConnected(64, 10))
    nn.add(Activation('softmax'))
    nn.add(SGDBatching(len(X_train)))
    nn.add(CrossEntropy())
    nn.add(AdamOptimizer(learning_rate=5e-3))
    nn.train(X_train, y_train, print_every=5, epochs=150, l2_alpha=.05, validation_fraction=0)
    print('Trained')
    assert ClassificationCost.accuracy(y_test, nn.eval(X_test)) >= .9
