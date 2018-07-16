# layers/lambda.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'Lambda'
]

from .layer import Layer

class Lambda(Layer):

    def __init__(self, forward_op, backward_op, copy=False):
        Layer.__init__(self, copy=copy)
        self.forward_op = forward_op
        self.backward_op = backward_op

    def _forward(self, X):
        return self.forward_op(X)

    def _backward(self, error):
        return self.backward_op(error)

    def get_parameters(self):
        return None # Non-parametric layer
