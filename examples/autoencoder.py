# -*- coding: utf-8 -*-
# autoencoder.py: Implement a denoising autoencoder from scratch
# author : Antoine Passemiers, Robin Petit

from bgd.nn import NeuralStack
from bgd.batch import SGDBatching
from bgd.cost import MSE
from bgd.layers import FullyConnected, Activation, Convolutional2D, GaussianNoise, Dropout
from bgd.initializers import GaussianInitializer, UniformInitializer
from bgd.optimizers import MomentumOptimizer, AdamOptimizer

import pickle
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


np.seterr(all='raise')

def create_autoencoder():
    nn = NeuralStack()
    nn.add(GaussianNoise(0, .2, clip=(0, 1)))
    nn.add(FullyConnected(28*28, 50))
    nn.add(Activation(function='tanh'))
    nn.add(FullyConnected(50, 50))
    nn.add(Activation(function='tanh'))
    nn.add(FullyConnected(50, 2))
    nn.add(Activation(function='tanh'))
    nn.add(FullyConnected(2, 50))
    nn.add(Activation(function='tanh'))
    nn.add(FullyConnected(50, 50))
    nn.add(Activation(function='tanh'))
    nn.add(FullyConnected(50, 28*28))

    optimizer = AdamOptimizer(learning_rate=.005)
    nn.add(optimizer)
    nn.add(MSE())
    return nn

mnist = fetch_mldata("MNIST original")
X = mnist.data / 255.
y = np.reshape(mnist.target, (mnist.target.shape[0], 1))
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25)

def get_trained_autoencoder():
    try:
        with open('autoencoder.pickle', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        nn = create_autoencoder()
        nn.add(SGDBatching(2048))
        nn.train(X_train, X_train, l2_alpha=.1, epochs=50, print_every=5)
        nn.batch_op = None
        with open('autoencoder.pickle', 'wb') as f:
            pickle.dump(nn, f)
        return nn

if __name__ == '__main__':
    nn = get_trained_autoencoder()
    f, axarr = plt.subplots(2,2)
    for i in range(2):
        img = X_train[i].reshape(28, 28)
        img_prime = nn.eval(img.reshape(1, 28**2)).reshape(28, 28)
        axarr[i, 0].imshow(img)
        axarr[i, 1].imshow(img_prime)
    plt.show()


