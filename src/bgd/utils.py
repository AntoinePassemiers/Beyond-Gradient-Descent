# -*- coding: utf-8 -*-
# utils.py
# author : Robin Petit, Antoine Passemiers

from datetime import datetime as dt

def log(txt, end='\n'):
    """Log a message on stdout"""
    print('[{}]\t{}'.format(now(), txt), end=end, flush=True)


def now():
    """ Return a string representing present daytime """
    return dt.now().strftime('%Y-%m-%d %T.%f')


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
