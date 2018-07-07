# operators.py
# author : Antoine Passemiers, Robin Petit

from bgd.layers import Layer

import numpy as np


class Operator(Layer):

    def __init__(self):
        Layer.__init__(self, copy=False, save_input=False, save_output=False)

    def get_parameters(self):
        return None # Non-parametric


Log = type(
    "Log",
    (Operator,),
    {
        "_forward": lambda self, X: np.log(X),
        "_backward": lambda self, X: 1. / X
    })

Exp = type(
    "Exp",
    (Operator,),
    {
        "_forward": lambda self, X: np.exp(X),
        "_backward": lambda self, X: np.exp(X)
    })
