""" This module contains all the exceptions that are specific
to BGD. """

# errors.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'RequiredComponentError', 'WrongComponentTypeError',
    'NonLearnableLayerError'
]

class RequiredComponentError(Exception):
    """ Exception raised when a :class:`bgd.nn.NeuralStack`
    hasn't been setup properly and at least a component is missing.
    """
    pass

class WrongComponentTypeError(Exception):
    """ Exception raised when a component of unrecognized type is
    attempted to be added to a :class:`bgd.nn.NeuralStack`.
    """
    pass

class NonLearnableLayerError(Exception):
    """ Exception raised to warn that an attempt to update
    parameter of a non-parametric layer has been attempted.
    """
    pass
