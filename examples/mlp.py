# -*- coding: utf-8 -*-
# mlp.py
# author : Antoine Passemiers, Robin Petit

from bgd.nn import NeuralStack
from bgd.batch import SGDBatching
from bgd.cost import CrossEntropy
from bgd.layers import FullyConnected, Activation, Flatten, Dropout
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


def train_mlp(use_lbfgs=True):
    mnist = fetch_mldata("MNIST original")
    X = (mnist.data / 255 - .5) * 2
    y = np.reshape(mnist.target, (mnist.target.shape[0], 1))
    X = X.reshape((X.shape[0], 28, 28, 1)) # New shape: (None, 28, 28, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1./7.)

    nn = NeuralStack()
    n_hidden = 500


    if use_lbfgs:
        nn.add(Flatten())
        nn.add(FullyConnected(28*28, n_hidden))
        nn.add(Activation(function='sigmoid'))
        nn.add(FullyConnected(n_hidden, 10))
        nn.add(Activation(function='softmax'))

        nn.add(CrossEntropy())
        nn.add(SGDBatching(256))
        adam = AdamOptimizer(learning_rate=0.007)
        nn.add(LBFGS(m=30, first_order_optimizer=adam))
    else:
        nn.add(Flatten())
        nn.add(FullyConnected(28*28, n_hidden))
        nn.add(Activation(function='sigmoid'))
        nn.add(FullyConnected(n_hidden, 10))
        nn.add(Activation(function='softmax'))

        nn.add(CrossEntropy())
        nn.add(SGDBatching(512))
        nn.add(AdamOptimizer(learning_rate=0.007))

    t0 = time()
    nn.train(X_train, y_train, l2_alpha=0.01, epochs=6, print_every=100)
    train_acc = accuracy_score(np.squeeze(y_train), nn.eval(X_train).argmax(axis=1))
    test_acc = accuracy_score(np.squeeze(y_test), nn.eval(X_test).argmax(axis=1))
    print("Training accuracy: %f" % train_acc)
    print("Test accuracy: %f" % test_acc)
    print("Optimization time: %.2f" % (time() - t0))


if __name__ == "__main__":
    train_mlp(use_lbfgs=False)
