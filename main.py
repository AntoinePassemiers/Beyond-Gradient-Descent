# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers, Robin Petit

from bgd.nn import NeuralStack
from bgd.layers import FullyConnected, Activation, Flatten, Convolutional2D, Dropout
from bgd.initializers import GaussianInitializer, UniformInitializer
from bgd.optimizers import MomentumOptimizer, AdamOptimizer

import numpy as np
from sklearn.datasets import fetch_mldata, load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

np.seterr(all='raise', over='warn', under='warn')
np.random.seed(0xCAFE)
print('')

dataset = 'digits'

if __name__ == '__main__':
    if dataset == 'mnist':
        mnist = fetch_mldata("MNIST original")
        #X = (mnist.data / 255 - .5) * 2
        X = np.asarray(mnist.data, dtype=np.float)
        y = np.reshape(mnist.target, (mnist.target.shape[0]))
        X = X.reshape((X.shape[0], 28, 28, 1)) # New shape: (None, 28, 28, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    elif dataset == 'digits':
        digits = load_digits()
        X = digits.images 
        y = digits.target
        X = X.reshape((X.shape[0], 8, 8, 1))
    
    nn = NeuralStack()
    nn.add(Convolutional2D([4, 4, 1], 32, strides=(1, 1)))
    nn.add(Activation('sigmoid'))
    #nn.add(Convolutional2D([3, 3, 32], 32, strides=(1, 1)))
    #nn.add(Activation('relu'))
    #nn.add(Convolutional2D([2, 2, 32], 64, strides=(1, 1)))
    #nn.add(Activation('relu'))
    nn.add(Flatten())
    nn.add(FullyConnected(800, 1024))
    nn.add(Activation('sigmoid'))
    nn.add(FullyConnected(1024, 10))
    nn.add(Activation('softmax'))
    
    optimizer = AdamOptimizer()
    nn.train(X, y, optimizer=optimizer, batch_size=len(X), epochs=1000, print_every=len(X), validation_fraction=0.25, alpha=.001)
    
    '''# digits
    nn = NeuralStack()
    nn.add(Convolutional2D([3, 3, 1], 32, strides=[1, 1]))
    nn.add(Flatten())
    nn.add(Activation('relu'))
    nn.add(FullyConnected(1152, len(np.unique(y))))
    nn.add(Activation('softmax'))
    nn.train(X, y, epochs=100, print_every=20000)'''

def test_mnist_mlp():
    mnist = fetch_mldata("MNIST original")
    X = (mnist.data / 255 - .5) * 2
    y = np.reshape(mnist.target, (mnist.target.shape[0], 1))
    X = X.reshape((X.shape[0], 28, 28, 1)) # New shape: (None, 28, 28, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    nn = NeuralStack()
    n_hidden = 512
    initializer_1 = GaussianInitializer(0, 1./28)
    initializer_2 = UniformInitializer(0, 1./n_hidden)
    # nn.add(Convolutional2D([3, 3, 1], 12, strides=[1, 1]))
    nn.add(Flatten())
    nn.add(FullyConnected(28*28, n_hidden, initializer=initializer_1))
    nn.add(Activation(function='relu'))
    nn.add(FullyConnected(n_hidden, 10, initializer=initializer_2))
    nn.add(Activation(function='softmax'))

    optimizer = AdamOptimizer()
    #optimizer = MomentumOptimizer(learning_rate=0.005, momentum=0.9)
    nn.train(X_train, y_train, optimizer=optimizer, batch_size=32, alpha=0.0001, epochs=2, print_every=100)
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