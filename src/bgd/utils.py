""" This module contains all the small and useful functions.
All the... utils... """

# utils.py
# author : Antoine Passemiers, Robin Petit

__all__ = [
    'log', 'now'
]

from datetime import datetime as dt

def log(txt, end='\n'):
    """Log a message on stdout"""
    print('[{}]\t{}'.format(now(), txt), end=end, flush=True)

def now():
    """ Return a string representing present daytime """
    return dt.now().strftime('%Y-%m-%d %T.%f')
