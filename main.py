from bgd.nn import FullyConnected, sigmoid, NeuralNet

import numpy as np

from sklearn.datasets import load_digits


import matplotlib.pyplot as plt

if __name__ == '__main__':
    nn = NeuralNet()

    digits = load_digits()
    X, y = digits.data / 255, np.reshape(digits.target, (digits.target.shape[0], 1))
    
    nn.add_layers((X.shape[1], 500, 10))
    errors = nn.train_gd(X, y, steps=1000)
    #plt.plot(errors)
    #plt.show()