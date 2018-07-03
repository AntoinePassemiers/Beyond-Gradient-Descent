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
    pass


class WrongComponentTypeError(Exception):
    pass

class NonLearnableLayerError(Exception):
    pass
