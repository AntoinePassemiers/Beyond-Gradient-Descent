""" This module contains all the layers that are implemented.
Any layer needs to inherit from :class:`bgd.layers.Layer` and
to implement its abstract methods (:obj:`_forward`, :obj:`backward`
and :obj:`get_parameters`, even by returning None if layer is
non-parametric). """

from .activation import *
from .conv2d import *
from .dropout import *
from .fc import *
from .flatten import *
from ._lambda import *
from .layer import *
from .max_pooling2d import *
from .noise import *
from .operators import *
