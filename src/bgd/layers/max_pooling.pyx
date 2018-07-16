# max_pooling.pyx
# author : Antoine Passemiers, Robin Petit
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.string cimport memset, memcpy

import ctypes
from cython.parallel import parallel, prange

from .conv cimport *


def max_pooling_2d_forward(data_t[:, :, :, :] output, cnp.int8_t[:, :, :, :] mask, data_t[:, :, :, :] X, object pool_shape, object strides):
    cdef int n_instances = X.shape[0]
    cdef int height = X.shape[1]
    cdef int width = X.shape[2]
    cdef int n_channels = X.shape[3]
    cdef int stride_h = strides[0]
    cdef int stride_w = strides[1]
    cdef int pool_height = pool_shape[0]
    cdef int pool_width = pool_shape[1]
    cdef int out_height = output.shape[1]
    cdef int out_width = output.shape[2]
    cdef int a, c, i, j, k, l, best_k, best_l
    cdef data_t max_el, el, MAX = <data_t> -np.inf
    np.asarray(mask)[:, :, :, :] = 0
    with nogil, parallel():
        for a in prange(n_instances):
            for i in range(out_height):
                for j in range(out_width):
                    for c in range(n_channels):
                        max_el = MAX
                        for k in range(pool_height):
                            for l in range(pool_width):
                                el = X[a, i*stride_h+k, j*stride_w+l, c]
                                if el > max_el:
                                    max_el, best_k, best_l = el, k, l
                        output[a, i, j, c] = max_el
                        mask[a, i*stride_h+best_k, j*stride_w+best_l, c] = 1


def max_pooling_2d_backward(data_t[:, :, :, :] output, data_t[:, :, :, :] error, cnp.int8_t[:, :, :, :] mask, object pool_shape, object strides):
    cdef int n_instances = output.shape[0]
    cdef int height = output.shape[1]
    cdef int width = output.shape[2]
    cdef int n_channels = output.shape[3]
    cdef int stride_h = strides[0]
    cdef int stride_w = strides[1]
    cdef int pool_height = pool_shape[0]
    cdef int pool_width = pool_shape[1]
    cdef int out_height = error.shape[1]
    cdef int out_width = error.shape[2]
    cdef int a, c, i, j, k, l
    np.asarray(output)[:, :, :, :] = 0
    with nogil, parallel():
        for a in prange(n_instances):
            for i in range(out_height):
                for j in range(out_width):
                    for c in range(n_channels):
                        for k in range(pool_height):
                            for l in range(pool_width):
                                if mask[a, i*stride_h+k, j*stride_w+l, c] > 0:
                                    output[a, i*stride_h+k, j*stride_w+l, c] = error[a, i, j, c]
