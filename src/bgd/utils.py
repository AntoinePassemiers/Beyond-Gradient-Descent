# -*- coding: utf-8 -*-
# utils.py
# author : Robin Petit, Antoine Passemiers

from datetime import datetime as dt


def log(txt, end='\n'):
    print('[{}]\t{}'.format(now(), txt), end=end, flush=True)


def now():
    return dt.now().strftime('%Y-%m-%d %T.%f')


class RequiredComponentError(Exception): pass


class WrongComponentTypeError(Exception): pass
