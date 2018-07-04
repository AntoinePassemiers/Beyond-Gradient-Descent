# -*- coding: utf-8 -*-
# nn.py
# author : Antoine Passemiers, Robin Petit

from bgd.batch import Batching
from bgd.cost import Cost, CrossEntropy
from bgd.errors import RequiredComponentError, WrongComponentTypeError
from bgd.layers import Layer, Dropout
from bgd.optimizers import Optimizer
from bgd.utils import log

import numpy as np

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
    if not (0 <= validation_fraction <= 1):
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

    def get_accuracy(self, X_val, y_val, batch_size=256):
        """ Returns the percentage of samples correctly labeled by the model.

        Args:
            X_val (:obj:`np.ndarray`):
                Samples to label.
            y_val (:obj:`np.ndarray`):
                Labels of the provided samples.
            batch_size (int):
                Number of samples from X_val to insert per batch.

        Returns:
            float:
                accuracy:
                    Percentage of correctly labeled samples (in [0, 100]).
        """
        nb_correct = 0
        for i in np.arange(0, X_val.shape[0], batch_size):
            indices = i + np.arange(min(batch_size, X_val.shape[0]-i))
            val_probs = self.eval(X_val[indices])
            assert len(val_probs) == len(indices)
            nb_correct += (val_probs.argmax(axis=1) == y_val[indices]).sum()
        return 100 * nb_correct / X_val.shape[0]

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

    def train(self, X, y, epochs=1000, alpha_reg=.0001,
              print_every=50, validation_fraction=0.1):
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
            alpha_reg (float):
                L2 regularization alpha parameter (0 if no L2).
            print_every (int):
                Number of instances between two prints of model state and
                evaluations on the intermediate validation set (negative
                number if none is wanted).
            validation_fraction (float):
                Proportion of samples to be kept for validation.

        Returns:
            :obj:`np.ndarray`:
                errors:
                    array of the loss of each batch.

        Raises:
            ValueError:
                see :func:`split_train_val`.

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
            Loss at epoch 93 (batch 1)       : 0.4746132147046453   - Validation accuracy: 95.0
            Loss at epoch 187 (batch 1)       : 0.31422001176102426  - Validation accuracy: 96.7
            Loss at epoch 281 (batch 1)       : 0.2320271116849095   - Validation accuracy: 95.6
            Loss at epoch 374 (batch 1)       : 0.2071083076653019   - Validation accuracy: 96.1
            Loss at epoch 468 (batch 1)       : 0.19040431125095156  - Validation accuracy: 95.6
            Loss at epoch 562 (batch 1)       : 0.170319350663334    - Validation accuracy: 96.1
            Loss at epoch 656 (batch 1)       : 0.1645264159422448   - Validation accuracy: 95.0
            Loss at epoch 749 (batch 1)       : 0.1481725308805749   - Validation accuracy: 95.6
            Loss at epoch 843 (batch 1)       : 0.1347951182675034   - Validation accuracy: 95.0
            Loss at epoch 937 (batch 1)       : 0.12705829279032987  - Validation accuracy: 95.0
            >>> print('Loss over time:', losses)
            Errors: [ 2.50539121  2.28391007  2.40779468 ...,  0.11655055  0.12436938  0.09155006]
        """
        losses = list()

        # Split data into training data and validation data for early stopping
        if validation_fraction > 0:
            X_train, y_train, X_val, y_val = split_train_val(X, y, validation_fraction)
        else:
            X_train, y_train = X, y

        # Binarize labels if classification task
        if isinstance(self.cost_op, CrossEntropy):
            y_train = binarize_labels(y_train)

        # Deactivate signal propagation though first layer
        self.layers[0].deactivate_propagation()

        seen_instances = 0
        for epoch in range(epochs):
            batch_id = 0
            self.batch_op.start(X_train, y_train)
            next_batch = self.batch_op.next()
            while next_batch:
                # Retrieve batch for next iteration
                (batch_x, batch_y) = next_batch
                next_batch = self.batch_op.next()

                # Forward pass
                predictions = self.eval(batch_x)

                # Compute loss function
                alpha = alpha_reg / self.batch_op.batch_size
                loss = self.eval_loss(batch_y, predictions, alpha_reg)
                losses.append(loss)

                # Compute gradient of the loss function
                signal = self.cost_op.grad(batch_y, predictions)

                # Propagate error through each layer
                for layer in reversed(self.layers):
                    extra_info = {'l2_reg': alpha}
                    signal, gradient = layer.backward(signal, extra_info)
                    if gradient is not None:
                        self.optimizer.add_gradient_fragments(layer, gradient)
                F = lambda: self.eval_loss(batch_y, self.eval(batch_x), alpha)
                self.optimizer.update(F)
                self.optimizer.flush()

                seen_instances += self.batch_op.batch_size
                if seen_instances % print_every == 0:
                    batch_id += 1
                    # Warning: This code section is ugly
                    if isinstance(self.cost_op, CrossEntropy):
                        if validation_fraction > 0:
                            val_accuracy = self.get_accuracy(X_val, y_val)
                        else:
                            val_accuracy = -1
                        log(('Loss at epoch {0} (batch {1: <9}:' + \
                             ' {2: <20} - Validation accuracy: {3:.1f}') \
                             .format(epoch, str(batch_id) + ')', loss, val_accuracy))
                    else:
                        if validation_fraction > 0:
                            val_preds = self.eval(X_val)
                            val_mse = self.cost_op.eval(y_val, val_preds)
                        else:
                            val_accuracy = -1
                        log(('Loss at epoch {0} (batch {1: <9}: ' + \
                             '{2: <20} - Validation MSE: {3: <15}') \
                             .format(epoch, str(batch_id) + ')', loss, val_mse))
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
        if stop == -1 or stop > len(self.layers):
            stop = len(self.layers)
        for i in range(start, stop):
            X = self.layers[i].forward(X)
        return X

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
