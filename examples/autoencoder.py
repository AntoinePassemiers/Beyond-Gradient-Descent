# -*- coding: utf-8 -*-
# autoencoder.py: Implement a denoising autoencoder from scratch
# author : Antoine Passemiers, Robin Petit

from bgd.nn import NeuralStack
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

mnist = fetch_mldata("MNIST original")
X = mnist.data / 255
y = np.reshape(mnist.target, (mnist.target.shape[0], 1))
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25)

initializer = GaussianInitializer(0, .1)

nn = NeuralStack()
nn.add(GaussianNoise(.2, clip=(0, 1)))
nn.add(FullyConnected(28*28, 50, initializer=initializer))
nn.add(Activation(function='tanh'))
nn.add(FullyConnected(50, 50, initializer=initializer))
nn.add(Activation(function='tanh'))
nn.add(FullyConnected(50, 2, initializer=initializer))
nn.add(Activation(function='tanh'))
nn.add(FullyConnected(2, 50, initializer=initializer))
nn.add(Activation(function='tanh'))
nn.add(FullyConnected(50, 50, initializer=initializer))
nn.add(Activation(function='tanh'))
nn.add(FullyConnected(50, 28*28, initializer=initializer))


optimizer = AdamOptimizer(learning_rate=.01)
nn.train(X_train, X_train, error_op='mse', optimizer=optimizer, batch_size=32, alpha=0.001, epochs=6, print_every=10)

f, axarr = plt.subplots(2,2)
for i in range(2):
    img = X_train[i].reshape(28, 28)
    img_prime = nn.eval(img.reshape(1, 28**2)).reshape(28, 28)
    axarr[i, 0].imshow(img)
    axarr[i, 1].imshow(img_prime)
plt.show()

pickle.dump(nn, open('autoencoder.pickle', 'wb'))