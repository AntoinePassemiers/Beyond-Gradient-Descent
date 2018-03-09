# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers, Robin Petit

from bgd.nn import NeuralStack
from bgd.layers import FullyConnected, Activation, Flatten, Convolutional2D, Dropout
from bgd.initializers import GaussianInitializer, UniformInitializer

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt



if __name__ == '__main__':

    np.random.seed(0)
    np.seterr(all='raise')

    mnist = fetch_mldata("MNIST original")
    X, y = mnist.data / 255, np.reshape(mnist.target, (mnist.target.shape[0], 1))
    X = X.reshape((X.shape[0], 28, 28, 1)) # New shape: (None, 28, 28, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    nn = NeuralStack()
    n_hidden = 512
    # As suggested by LeCun in 1998 for weights initialization
    initializer_1 = GaussianInitializer(0, 1./28)
    initializer_2 = UniformInitializer(0, 1./n_hidden)
    
    # nn.add(Convolutional2D([3, 3, 1], 12, strides=[1, 1]))
    nn.add(Flatten())
    # nn.add(Dropout())
    nn.add(FullyConnected(28*28, n_hidden, initializer=initializer_1))
    nn.add(Activation(function='sigmoid'))
    nn.add(FullyConnected(n_hidden, 10, initializer=initializer_2))
    nn.add(Activation(function='softmax'))

    nn.train(X_train, y_train, batch_size=32, alpha=0.0001, epochs=2, learning_rate=0.01, print_every=100, momentum=0.5)
    train_acc = accuracy_score(np.squeeze(y_train), nn.eval(X_train).argmax(axis=1))
    test_acc = accuracy_score(np.squeeze(y_test), nn.eval(X_test).argmax(axis=1))
    print("Training accuracy: %f" % train_acc)
    print("Test accuracy: %f" % test_acc)

    """
    C = confusion_matrix(np.squeeze(y_test), nn.eval(X_test).argmax(axis=1))
    plt.imshow(C)
    plt.colorbar()
    plt.title('Confusion matrix')
    plt.show()
    """