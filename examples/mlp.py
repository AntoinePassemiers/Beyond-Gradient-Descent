# -*- coding: utf-8 -*-
# mlp.py
# author : Antoine Passemiers, Robin Petit

from bgd.nn import NeuralStack
from bgd.batch import SGDBatching
from bgd.errors import CrossEntropy
from bgd.layers import FullyConnected, Activation, Flatten, Dropout
from bgd.initializers import GaussianInitializer, UniformInitializer
from bgd.optimizers import MomentumOptimizer, AdamOptimizer, LBFGS
from bgd.utils import log

from time import time
import numpy as np
from sklearn.datasets import fetch_mldata, load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from random import seed

np.seterr(all='raise', over='warn', under='warn')
np.random.seed(0xCAFE)
seed(0xCAFE)
print('')


if __name__ == "__main__":

    mnist = fetch_mldata("MNIST original")
    X = (mnist.data / 255 - .5) * 2
    y = np.reshape(mnist.target, (mnist.target.shape[0], 1))
    X = X.reshape((X.shape[0], 28, 28, 1)) # New shape: (None, 28, 28, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./7.)

    nn = NeuralStack()
    n_hidden = 200
    initializer_1 = GaussianInitializer(0, 1./28)
    initializer_2 = UniformInitializer(0, 1./n_hidden)

    nn.add(Flatten())
    nn.add(FullyConnected(28*28, n_hidden, initializer=initializer_1))
    nn.add(Activation(function='sigmoid'))
    nn.add(FullyConnected(n_hidden, 10, initializer=initializer_2))
    nn.add(Activation(function='softmax'))

    nn.add(CrossEntropy())
    nn.add(SGDBatching(512))

    adam = AdamOptimizer()
    #nn.add(adam)
    nn.add(LBFGS(m=20, first_order_optimizer=adam))

    nn.train(X_train, y_train, alpha_reg=0.0001, epochs=4, print_every=100)
    train_acc = accuracy_score(np.squeeze(y_train), nn.eval(X_train).argmax(axis=1))
    test_acc = accuracy_score(np.squeeze(y_test), nn.eval(X_test).argmax(axis=1))
    print("Training accuracy: %f" % train_acc)
    print("Test accuracy: %f" % test_acc)
