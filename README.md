[![Travis-CI Build Status](https://travis-ci.org/AntoinePassemiers/Beyond-Gradient-Descent.svg?branch=master)](https://travis-ci.org/AntoinePassemiers/Beyond-Gradient-Descent)
# BGD - Second order optimization for neural networks

Quick Example
-------------

Create an empty stack of layers.

```python
from bgd.nn import NeuralStack

nn = NeuralStack()
```

Define a two-layer perceptron architecture
with sigmoidal activation functions.

```python
from bgd.layers import FullyConnected, Activation

n_hidden = 500
nn.add(FullyConnected(784, n_hidden))
nn.add(Activation(function='sigmoid'))
nn.add(FullyConnected(n_hidden, 10))
nn.add(Activation(function='softmax'))
```

Set loss function, batching method and optimizer.
All components that are not layers can be added
at any stage of the stack construction in any order.
Here, a batch size of 512 samples is used.
Setting a batch size smaller than training set size
and using a first-order optimizer enables Stochastic
Gradient Descent.

```python
from bgd.batch import SGDBatching
from bgd.cost import CrossEntropy
from bgd.optimizers import AdamOptimizer

nn.add(CrossEntropy())
nn.add(SGDBatching(512))
nn.add(AdamOptimizer(learning_rate=0.007))
```

Train the model.

```python
nn.train(X_train, y_train, alpha_reg=0.0001, epochs=6, print_every=100)
```

Assess performance.

```python
train_acc = accuracy_score(np.squeeze(y_train), nn.eval(X_train).argmax(axis=1))
test_acc = accuracy_score(np.squeeze(y_test), nn.eval(X_test).argmax(axis=1))
print("Training accuracy: %f" % train_acc)
print("Test accuracy: %f" % test_acc)
```

More examples can be found in `examples/` folder.

Components
----------

Here is a list of components that are currently available. Full documentation is available
[here](https://antoinepassemiers.github.io/Beyond-Gradient-Descent/).

### Layers

* FullyConnected
* Activation
* Convolutional2D (NumPy and SSE implementations)
* MaxPooling2D
* Dropout
* GaussianNoise
* Flatten
* Lambda

### Optimizers

* MomentumOptimizer
* AdamOptimizer
* LBFGS

### Error operators

* MSE
* CrossEntropy

### Weight initializers

* ZeroInitializer
* UniformInitializer
* GlorotUniformInitializer
* GaussianInitializer
* GlorotGaussianInitializer


Installation
------------

### Dependencies

All the dependencies are listed in `requirements.txt`. To install them all, run:
```
$ pip3 install -r requirements.txt
```

### User installation

To install the package:

* if you terminal supports `make`, then run `make install` in the root;
* otherwise (or if you prefer), move to `src/` with `cd src` and then install
  the package with `python3 setup.py install`.

