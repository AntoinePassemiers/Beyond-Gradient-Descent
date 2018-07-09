# test.py
# author : Antoine Passemiers, Robin Petit

import numpy as np
from bgd.layers import Activation

def test_useless():
    assert(True)


def test_dependencies():
    a = np.empty(50)
    assert(Activation().forward(a) is not None)