# -*- coding: utf-8 -*-
# operators.pyx
# author : Antoine Passemiers, Robin Petit
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdlib cimport calloc


cdef fused data_t:
    cnp.int_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.float_t
    cnp.float32_t
    cnp.float64_t
    

def test():
    with nogil:
        pass
    print("Mangez des pommes. La pomme est un fruit... sympathique, je l'observe tous les jours.")


def conv_2d_forward(data_t[:, :, :, :] X, data_t[:, :, :, :] filters, data_t[:] b, object strides, bint add_bias):
    cdef Py_ssize_t a, c, f, i, j, k, l
    cdef cnp.int_t[:] c_strides = np.asarray(strides, dtype=np.int)
    cdef Py_ssize_t n_instances = X.shape[0]
    cdef Py_ssize_t height = X.shape[1]
    cdef Py_ssize_t width = X.shape[2]
    cdef Py_ssize_t n_channels = X.shape[3]
    cdef Py_ssize_t n_filters = filters.shape[0]
    cdef Py_ssize_t filter_height = filters.shape[1]
    cdef Py_ssize_t filter_width = filters.shape[2]
    cdef Py_ssize_t out_height = (height - filter_height + 1) // c_strides[0]
    cdef Py_ssize_t out_width = (width - filter_width + 1) // c_strides[1]
    cdef data_t[:, :, :, :] output = <data_t[:n_instances, :out_height, :out_width, :n_filters]>calloc(
        n_instances * out_height * out_width * n_filters, sizeof(data_t))
    cdef data_t temp
    with nogil:
        for a in range(n_instances):
            for i in range(out_height):
                for j in range(out_width):
                    for c in range(n_channels):
                        for f in range(n_filters):
                            temp = 0
                            for k in range(filter_height):
                                for l in range(filter_width):
                                    temp += filters[f, k, l, c]
                            output[a, i, j, f] += temp * X[a, i*c_strides[0], j*c_strides[1], c]
                        if add_bias:
                            output[a, i, j, f] += b[f] # Add intercept
    return np.asarray(output)