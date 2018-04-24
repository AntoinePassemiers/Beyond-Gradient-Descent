# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers, Robin Petit

from bgd.nn import NeuralStack
from bgd.layers import FullyConnected, Activation, Flatten, Convolutional2D, MaxPooling2D, Dropout
from bgd.initializers import GaussianInitializer, UniformInitializer
from bgd.optimizers import MomentumOptimizer, AdamOptimizer
from bgd.utils import log

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

dataset = 'mnist'

from time import time
from bgd.operators import conv_2d_forward_sse, conv_2d_forward

def compare_convolution(nb_rep: int=100):
    size = (32, 32, 32, 32)
    kernel_size = (16, 3, 3, 32)
    strides = [1, 1]
    biases = np.random.rand(kernel_size[0])
    array = np.random.rand(*size)
    kernel = np.random.rand(*kernel_size)
    output_array = np.empty((size[0], size[1]-kernel_size[1]+1, size[2]-kernel_size[2]+1, kernel_size[0]), dtype=np.float32)
    for n_jobs in np.arange(4)+1:
        min_time = np.inf
        max_time = -np.inf
        start = time()
        for _ in range(nb_rep):
            step_start = time()
            conv_2d_forward(output_array, array.astype(np.float32), kernel.astype(np.float32), biases.astype(np.float32), strides, True, n_jobs)
            step_time = time()-step_start
            if step_time > max_time:
                max_time = step_time
            if step_time < min_time:
                min_time = step_time
        end = time()
        print(('No SSE (n_jobs == {:d}): {:g} seconds for {:d} convolutions\n' + \
               '\ti.e. {:g} sec/convolution. (max, min) == ({:g}, {:g})') \
               .format(n_jobs, end-start, nb_rep, (end-start)/nb_rep, max_time, min_time))
    min_time = np.inf
    max_time = -np.inf
    start = time()
    for _ in range(nb_rep):
        step_start = time()
        conv_2d_forward_sse(output_array, array.astype(np.float32), kernel.astype(np.float32), biases.astype(np.float32), strides, True)
        step_time = time()-step_start
        if step_time > max_time:
            max_time = step_time
        if step_time < min_time:
            min_time = step_time
    end = time()
    print(('With SSE: {:g} seconds for {:d} convolutions\n' + \
           '\ti.e. {:g} sec/convolution. (max, min) == ({:g}, {:g})') \
           .format(end-start, nb_rep, (end-start)/nb_rep, max_time, min_time))

if __name__ == '__main__':
    compare_convolution()
    #test_cnn(dataset)
    
def test_cnn(dataset: str):
    if dataset == 'mnist':
        mnist = fetch_mldata("MNIST original")
        X = mnist.data / 255
        y = np.reshape(mnist.target, (mnist.target.shape[0]))
        X = X.reshape((X.shape[0], 28, 28, 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=512)
        batch_size = 512

        optimizer = AdamOptimizer(learning_rate=.005)

        nn = NeuralStack()
        nn.add(Convolutional2D([5, 5, 1], 32, strides=(2, 2)))
        nn.add(Activation('relu'))
        #nn.add(MaxPooling2D([2, 2], strides=[2, 2]))
        nn.add(Convolutional2D([5, 5, 32], 32, strides=(2, 2)))
        nn.add(Activation('relu'))
        #nn.add(MaxPooling2D([2, 2], strides=[2, 2]))
        #nn.add(Dropout(.9))
        nn.add(Flatten())
        #nn.add(FullyConnected(288, 64))
        nn.add(FullyConnected(512, 64))
        nn.add(Activation('sigmoid'))
        nn.add(FullyConnected(64, 10))
        nn.add(Activation('softmax'))

    elif dataset == 'digits':
        digits = load_digits()
        X = digits.images
        y = digits.target
        X = X.reshape((X.shape[0], 8, 8, 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        batch_size = len(X_train)

        optimizer = AdamOptimizer(learning_rate=.001)

        nn = NeuralStack()
        nn.add(Convolutional2D([4, 4, 1], 32, strides=(1, 1)))
        nn.add(Activation('relu'))
        #nn.add(Convolutional2D([3, 3, 32], 32, strides=(1, 1)))
        #nn.add(Activation('relu'))
        #nn.add(Convolutional2D([2, 2, 32], 64, strides=(1, 1)))
        #nn.add(Activation('relu'))
        nn.add(Flatten())
        nn.add(FullyConnected(800, 200))
        nn.add(Activation('relu'))
        nn.add(FullyConnected(200, 10))
        nn.add(Activation('softmax'))


    errors = nn.train(X_train, y_train, optimizer=optimizer, batch_size=batch_size,
             epochs=10, print_every=1*batch_size, validation_fraction=0.0, alpha=.1)
    accuracy_test = nn.get_accuracy(X_test, y_test, batch_size=1024)
    log('Accuracy on test: {:.3f}%'.format(accuracy_test))

    '''# digits
    nn = NeuralStack()
    nn.add(Convolutional2D([3, 3, 1], 32, strides=[1, 1]))
    nn.add(Flatten())
    nn.add(Activation('relu'))
    nn.add(FullyConnected(1152, len(np.unique(y))))
    nn.add(Activation('softmax'))
    errors = nn.train(X, y, epochs=100, print_every=20000)'''

    plt.plot(errors)
    plt.show()

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
