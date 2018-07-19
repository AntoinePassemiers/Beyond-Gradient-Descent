# -*- coding: utf-8 -*-
# cnn.py
# author : Antoine Passemiers, Robin Petit

from bgd.nn import NeuralStack
from bgd.batch import SGDBatching
from bgd.cost import CrossEntropy
from bgd.layers import FullyConnected, Activation, Flatten, Convolutional2D, MaxPooling2D, Dropout
from bgd.initializers import GaussianInitializer, UniformInitializer
from bgd.layers.conv import conv_2d_forward_sse, conv_2d_forward
from bgd.optimizers import MomentumOptimizer, AdamOptimizer
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

dataset = 'digits'


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


def test_cnn(dataset: str):
    if dataset == 'mnist':
        mnist = fetch_mldata("MNIST original")
        X = mnist.data / 255
        y = np.reshape(mnist.target, (mnist.target.shape[0]))
        X = X.reshape((X.shape[0], 28, 28, 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=512)
        batch_size = 512
        epochs = 10

        optimizer = AdamOptimizer(learning_rate=.005)

        nn = NeuralStack()
        nn.add(Convolutional2D([5, 5, 1], 32, strides=(2, 2)))
        nn.add(Activation('relu'))
        nn.add(Convolutional2D([5, 5, 32], 32, strides=(2, 2)))
        nn.add(Activation('relu'))
        nn.add(Flatten())
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

        optimizer = AdamOptimizer(learning_rate=.001)

        nn = NeuralStack()
        nn.add(Convolutional2D([4, 4, 1], 32, strides=(2, 2)))
        nn.add(Activation('relu'))
        #nn.add(Convolutional2D([3, 3, 32], 32, strides=(1, 1)))
        #nn.add(Activation('relu'))
        #nn.add(Convolutional2D([2, 2, 32], 64, strides=(1, 1)))
        #nn.add(Activation('relu'))
        nn.add(Flatten())
        nn.add(FullyConnected(288, 200))
        nn.add(Activation('relu'))
        nn.add(FullyConnected(200, 10))
        nn.add(Activation('softmax'))

        batch_size = 1024
        epochs = 50

    nn.add(optimizer)
    nn.add(CrossEntropy())
    nn.add(SGDBatching(batch_size))

    errors = nn.train(
        X_train, y_train, epochs=epochs, print_every=1,
        validation_fraction=0.0, l2_alpha=.005)
    accuracy_test = nn.get_accuracy(X_test, y_test)
    log('Accuracy on test: {:.3f}%'.format(accuracy_test))

    print('Last errors:', errors[-5:])
    plt.plot(errors)
    plt.show()


if __name__ == '__main__':
    #compare_convolution()
    test_cnn(dataset)
