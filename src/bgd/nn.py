""" This module contains the :class:`bgd.nn.NeuralStack`
class that represents a linear neural network (LNN).

Any other model of neural nets shall be written down here. """

# nn.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'split_train_val', 'binarize_labels', 'NeuralStack'
]

import numpy as np

from bgd.batch import Batching
from bgd.cost import ClassificationCost, Cost
from bgd.errors import RequiredComponentError, WrongComponentTypeError
from bgd.layers import Layer, Dropout
from bgd.optimizers import Optimizer
from bgd.utils import log

def split_train_val(X, y, validation_fraction):
    """ Splits randomly the dataset (X, y) into a training set
    and a validation set.

    Args:
        X (:obj:`np.ndarray`):
            Matrix of samples. shape == (n_instances, n_features).
        y (:obj:`np.ndarray`):
            Vector of expected outputs. shape == (n_instances,)
            or (n_instances, n_classes) if binarized.
        validation_fraction (float):
            Proportion of samples to be kept for validation.

    Returns:
        :
            X_train (:obj:`np.ndarray`):
                Matrix of training samples.
                len(X_train) == len(X) * (1 - validation_fraction)
            y_train (:obj:`np.ndarray`):
                Vector of training expected outputs.
                len(y_train) == len(y) * (1 - validation_fraction)
            X_test (:obj:`np.ndarray`):
                Matrix of test samples.
                len(X_test) == len(X) * validation_fraction
            y_test (:obj:`np.ndarray`):
                Vector of test expected outputs.
                len(y_test) == len(y) * validation_fraction

    Raises:
        ValueError:
            If validation_fraction is not in [0, 1].
    """
    if not 0 <= validation_fraction <= 1:
        raise ValueError('{:g} is not a valid fraction'.format(validation_fraction))
    split = int(len(X) * (1. - validation_fraction))
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], np.squeeze(y[indices])
    return X[:split], y[:split], X[split:], y[split:]

def binarize_labels(y):
    """ Transforms a N-valued vector of labels into a binary vector.
    If N classes are present, they *must* be 0 to N-1.

    Args:
        y (:obj:`np.ndarray`):
            Vector of classes.

    Returns:
        :obj:`np.ndarray`:
            binary_y:
                Binary matrix of shape (n_instances, n_classes)
                s.t. `binary_y[i,j] == 1` iff `y[i] == j`.

    Example:
        >>> y = np.array([0, 1, 0, 0, 1, 1, 0])
        >>> binarize_labels(y)
        array([[1, 0],
               [0, 1],
               [1, 0],
               [1, 0],
               [0, 1],
               [0, 1],
               [1, 0]])
    """
    unique_values = np.unique(y)
    n_classes = len(unique_values)
    if set(unique_values) != set(np.arange(n_classes)):
        raise ValueError('The {} classes must be encoded 0 to {}' \
                         .format(n_classes, n_classes-1))
    binary_y = np.zeros((len(y), n_classes), dtype=np.int)
    for c in range(n_classes):
        binary_y[:, c] = (y == c)
    return binary_y


class NeuralStack:
    """ Sequential model that can be viewed as a stack of layers.
    Backpropagation is simplified because the error gradient
    with respect to the parameters of any layer is the gradient
    of a composition of functions (defined by all successor layers)
    and is decomposed using the chain rule.

    Attributes:
        layers (list):
            List of layers, where layers[0] is the input layer and
            layers[-1] is the output layer.
        batch_op (:class:`bgd.batch.Batching`):
            Batching method to use in order to perform one step of the
            backpropagation algorithm.
        cost_op (:class:`bgd.cost.Cost`):
            Error metric to use in order to evaluate model performance.
        optimizer (:class:`bgd.optimizers.Optimizer`):
            Optimizer to use in order to update the parameters of the
            model during backpropagation.
    """

    def __init__(self):
        self.layers = list()
        self.batch_op = None
        self.cost_op = None
        self.optimizer = None

    def add(self, component):
        """ Adds a component to the model: it can be either a layer
        or a special component like an optimizer. Some components
        are mandatory to train the model. The architecture of the
        model is defined by the layers that are provided to this
        method. Thus, the order is taken into account when adding
        layers. However, the order has no importance when adding
        special components like optimizers, error metrics, etc.

        Args:
            component (object):
                Component to add to the model (Layer or special component).

        Raises:
            WrongComponentTypeError:
                If the type of component is not recognized.
        """
        if isinstance(component, Layer):
            # Add a layer to the neural stack
            self.layers.append(component)
        elif isinstance(component, Optimizer):
            # Set the optimizer
            self.optimizer = component
        elif isinstance(component, Cost):
            # Set the error metric
            self.cost_op = component
        elif isinstance(component, Batching):
            # Set the batching algorithm
            self.batch_op = component
        else:
            raise WrongComponentTypeError("Unknown component type")

    def eval_loss(self, batch_y, predictions, alpha):
        """ Returns the loss value of the output computed by the model
        with respect to the actual output.

        Args:
            batch_y (:obj:`np.ndarray`):
                True labels of the samples.
            predictions (:obj:`np.ndarray`):
                Labels predicted by the model.
            alpha (float):
                L2 regularization alpha term (0 if no L2).

        Returns:
            float:
                loss:
                    The value of the loss function.
        """
        loss = self.cost_op.eval(batch_y, predictions)
        # Apply L2 regularization
        if alpha > 0:
            for layer in self.layers:
                params = layer.get_parameters()
                if params:
                    squared_params = params[0] ** 2
                    if not np.isnan(squared_params).any():
                        loss += 0.5 * alpha * np.sum(squared_params)
        return loss

    def train(self, X, y, epochs=1000, l2_alpha=.1,
              print_every=1, validation_fraction=0.1,
              dataset_normalization=False):
        """ Trains the model on samples X and labels y. Optimize the model
        parameters so that the loss is minimized on this dataset.
        The validation fraction *must* be in [0, 1] (0 to not evaluate the
        model on a validation set).

        Args:
            X (:obj:`np.ndarray`):
                Array of samples. `shape == (n_instances, n_features)` or
                `shape == (n_instances, n_pixels, n_channels)` for images.
            y (:obj:`np.ndarray`):
                Array of labels. `shape == (n_instances,)`
            epochs (int):
                Number of times the dataset (X, y) is entirely fed to the model.
            l2_alpha (float):
                L2 regularization alpha parameter (0 if no L2).
            print_every (int):
                Number of batches between two prints of model state and
                evaluations on the intermediate validation set (negative
                number if none is wanted). Defaults to 1.
            validation_fraction (float):
                Proportion of samples to be kept for validation.

        Returns:
            :obj:`np.ndarray`:
                errors:
                    array of the loss of each batch.

        Raises:
            ValueError:
                see :func:`split_train_val`.
            RequiredComponentError:
                see :meth:`check_components`.

        Example:
            >>> X, y = load_digits(return_X_y=True)
            >>> n_in, n_hidden, n_out = 64, 32, 10
            >>> nn = NeuralStack()
            >>> nn.add(FullyConnected(n_in, n_hidden))
            >>> nn.add(Activation())
            >>> nn.add(FullyConnected(n_hidden, n_out))
            >>> nn.add(Activation('softmax'))
            >>> nn.add(CrossEntropy())
            >>> nn.add(AdamOptimizer())
            >>> nn.add(SGDBatching(512))
            >>> losses = nn.train(X, y)
            Loss at epoch 49 (batch 2): 1.2941039726880612   - Validation accuracy: 83.333%
            Loss at epoch 99 (batch 2): 0.764482839362229    - Validation accuracy: 96.111%
            Loss at epoch 149 (batch 2): 0.5154527368075816   - Validation accuracy: 97.222%
            Loss at epoch 199 (batch 2): 0.3979498104086121   - Validation accuracy: 97.778%
            Loss at epoch 249 (batch 2): 0.32569535096131924  - Validation accuracy: 97.778%
            Loss at epoch 299 (batch 2): 0.2748402661346476   - Validation accuracy: 97.778%
            Loss at epoch 349 (batch 2): 0.2555267910622358   - Validation accuracy: 97.778%
            Loss at epoch 399 (batch 2): 0.22863001357754167  - Validation accuracy: 97.778%
            Loss at epoch 449 (batch 2): 0.22746067257186584  - Validation accuracy: 97.778%
            Loss at epoch 499 (batch 2): 0.22220948073377536  - Validation accuracy: 97.778%
            Loss at epoch 549 (batch 2): 0.2089401746378744   - Validation accuracy: 98.333%
            Loss at epoch 599 (batch 2): 0.20035077161054704  - Validation accuracy: 98.333%
            Loss at epoch 649 (batch 2): 0.1867468456143161   - Validation accuracy: 98.333%
            Loss at epoch 699 (batch 2): 0.17347271604108705  - Validation accuracy: 98.333%
            Loss at epoch 749 (batch 2): 0.15376433332896908  - Validation accuracy: 98.333%
            Loss at epoch 799 (batch 2): 0.15436015140582668  - Validation accuracy: 98.333%
            Loss at epoch 849 (batch 2): 0.13860411664942396  - Validation accuracy: 98.889%
            Loss at epoch 899 (batch 2): 0.133165375570591    - Validation accuracy: 98.333%
            Loss at epoch 949 (batch 2): 0.12890010110436428  - Validation accuracy: 98.333%
            Loss at epoch 999 (batch 2): 0.12433454132628034  - Validation accuracy: 98.333%
            >>> print('Loss over time:', losses)
            Errors: [ 2.50539121  2.28391007  2.40779468 ...,  0.11655055  0.12436938  0.09155006]
        """
        dtype = np.float32  # Force the NN to work only with np.float32
        # Check arrays
        X, y = np.asarray(X), np.asarray(y)
        if X.dtype is not dtype:
            X = X.astype(dtype)
        if dataset_normalization:
            X = (X - X.mean()) / X.std()

        # Check components
        self.check_components()

        # Split data into training data and validation data for early stopping
        if len(X) < 50:
            validation_fraction = 0
        if validation_fraction > 0:
            X_train, y_train, X_val, y_val = split_train_val(X, y, validation_fraction)
        else:
            X_train, y_train = X, y

        # Binarize labels if classification task
        if isinstance(self.cost_op, ClassificationCost):
            y_train = binarize_labels(y_train)

        # Deactivate signal propagation though first layer
        self.layers[0].deactivate_propagation()

        losses = list()
        nb_batches = 0
        if l2_alpha < 0:
            l2_alpha = 0
        else:
            l2_alpha /= self.batch_op.batch_size  # TODO: batch_size is not an attribute of Batching
        for epoch in range(epochs):
            batch_id = 0
            self.batch_op.start(X_train, y_train)
            next_batch = self.batch_op.next()
            while next_batch:
                batch_x, batch_y = next_batch
                batch_id += 1
                nb_batches += 1

                # Forward pass
                predictions = self.eval(batch_x)

                # Compute loss function
                loss = self.eval_loss(batch_y, predictions, l2_alpha)
                losses.append(loss)

                # Compute gradient of the loss function
                signal = self.cost_op.grad(batch_y, predictions).astype(dtype)

                # Propagate error through each layer
                for layer in reversed(self.layers):
                    signal, gradient = layer.backward(signal)
                    if gradient is not None:
                        self.optimizer.add_gradient_fragments(layer, gradient, l2_alpha)
                F = lambda: self.eval_loss(batch_y, self.eval(batch_x), l2_alpha)  # pylint: disable=cell-var-from-loop
                self.optimizer.update(F)

                if print_every > 0 and nb_batches % print_every == 0:
                    log('Loss at epoch {} (batch {}): {: <20}' \
                        .format(epoch, batch_id, loss),
                        end=' - Validation ' if validation_fraction > 0 else '\n')
                    if validation_fraction > 0:
                        self.cost_op.print_fitness(y_val, self.eval(X_val))

                # Retrieve batch for next iteration
                next_batch = self.batch_op.next()
        return np.array(losses)

    def eval(self, X, start=0, stop=-1):
        """ Feeds a sample (or batch) to the model linearly from any
        layer to any layer. By default, X is propagated through the
        entire stack.

        Args:
            X (:obj:`np.ndarray`):
                The sample (or batch of samples) to feed to the model.
            start (int):
                Index of the layer where X is to be fed.
            stop (int):
                Index of the last layer of propagation (-1 for last layer).

        Returns:
            :obj:`np.ndarray`:
                out:
                    Output of the propagation of X through each layer
                    of the model.
        """
        X = np.asarray(X, dtype=np.float32)
        if stop == -1 or stop > len(self.layers):
            stop = len(self.layers)
        for i in range(start, stop):
            X = self.layers[i].forward(X)
        return X

    def get_accuracy(self, X, y):
        """ Returns the accuracy of the provided dataset on the model.

        Raises:
            :class:`AttributeError` if the model is not classifying.

        See :meth:`bgd.ClassificationCost.accuracy`.
        """
        if isinstance(self.cost_op, ClassificationCost):
            #if y.ndim == 1:
            #    y = binarize_labels(y)
            return ClassificationCost.accuracy(y, self.eval(X))
        raise AttributeError('Model is not classifying. Accuracy unavailable.')

    def activate_dropout(self):
        """ Activates the Dropout layers (for training phase). """
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.activate()

    def deactivate_dropout(self):
        """ Deactivates the Dropout layers (after training phase). """
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.deactivate()

    def check_components(self):
        """ Verifies that all the components are set properly so
        that training is possible. If either the batch method, the
        optimizer or the error function is missing, an exception
        will be raised.

        Raises:
            RequiredComponentError:
                If any component of the model has not been set.
        """
        # Check batch optimization method
        if self.batch_op is None:
            raise RequiredComponentError(Batching.__name__)

        # Check optimizer
        if self.optimizer is None:
            raise RequiredComponentError(Optimizer.__name__)

        # Check loss function
        if self.cost_op is None:
            raise RequiredComponentError(Cost.__name__)
