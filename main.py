# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers, Robin Petit

from bgd.nn import NeuralStack
from bgd.layers import FullyConnected, Activation, Flatten, Convolutional2D, Dropout
from bgd.initializers import GaussianInitializer

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


if __name__ == '__main__':

    np.random.seed(0)

    mnist = fetch_mldata("MNIST original")
    X, y = mnist.data / 255, np.reshape(mnist.target, (mnist.target.shape[0], 1))
    X = X.reshape((X.shape[0], 28, 28, 1)) # New shape: (None, 28, 28, 1)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    nn = NeuralStack()
    initializer = GaussianInitializer(0, 1)
    
    # nn.add(Convolutional2D([3, 3, 1], 12, strides=[1, 1]))
    nn.add(Flatten())
    nn.add(FullyConnected(28*28, 200, initializer=initializer))
    nn.add(Activation(function='sigmoid'))
    # nn.add(Dropout())
    nn.add(FullyConnected(200, 10, initializer=initializer))
    nn.add(Activation(function='softmax'))

    errors = nn.train(X_train, y_train, batch_size=200, alpha=0.0001, steps=10, learning_rate=0.1, print_every=500)

    
    plt.plot(errors)
    plt.ylabel("MSE")
    plt.xlabel("Iterations")
    plt.show()
    
    train_acc = accuracy_score(np.squeeze(y_train), nn.eval(X_train).argmax(axis=1))
    test_acc = accuracy_score(np.squeeze(y_test), nn.eval(X_test).argmax(axis=1))
    print("Training accuracy: %f" % train_acc)
    print("Test accuracy: %f" % test_acc)