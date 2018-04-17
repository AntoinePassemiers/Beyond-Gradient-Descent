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

import ctypes
from cython.parallel import parallel, prange


cdef fused data_t:
    #cnp.int_t
    #cnp.int8_t
    #cnp.int16_t
    #cnp.int32_t
    #cnp.int64_t
    cnp.float_t
    cnp.float32_t
    cnp.float64_t


# TODO: add padding

### Needs to be debugged after weight update has been corrected
def conv_2d_backward(data_t[:, :, :, :] output, data_t[:, :, :, :] epsilon, data_t[:, :, :, :] filters, object strides, int num_threads):
    cdef cnp.int_t[:] c_strides = np.asarray(strides, dtype=np.int)
    cdef int n_instances = epsilon.shape[0]
    cdef int height = epsilon.shape[1]
    cdef int width = epsilon.shape[2]
    cdef int n_channels = epsilon.shape[3]
    cdef int filter_height = filters.shape[1]
    cdef int filter_width = filters.shape[2]
    cdef int n_filters = filters.shape[3]
    cdef int out_height = (height + filter_height - 1) // c_strides[0]
    cdef int out_width = (width + filter_width - 1) // c_strides[1]
    cdef int instance, i, j, f, h, w, c, alpha, beta, h_0, w_0
    np.asarray(output)[:, :, :, :] = 0
    with nogil, parallel(num_threads=num_threads):
        # \alpha_0
        for instance in prange(n_instances):
            # \alpha_1
            for i in range(out_height):
                # \alpha_2
                for j in range(out_width):
                    ##h_0 = 0 if i <= filter_height else 1 + (i-filter_height) // c_strides[0]
                    # \beta_1
                    ##for h in range(h_0, min(height, i+1)):
                    for h in range(height):
                        if 0 <= i-c_strides[0]*h < filter_height:
                            ##w_0 = 0 if j <= filter_width else j-filter_width+1
                            # \beta_2
                            ##for w in range(w_0, min(width, j+1)):
                            for w in range(width):
                                if 0 <= j-c_strides[1]*w < filter_width:
                                    # \alpha_3
                                    for f in prange(n_filters):
                                        # \beta_3
                                        for c in prange(n_channels):
                                            output[instance, i, j, f] += epsilon[instance, h, w, c] * filters[c, i-c_strides[0]*h, j-c_strides[1]*w, f]


def conv_2d_backward_weights(data_t[:,:,:,:] output, data_t[:,:,:,:] X, data_t[:,:,:,:] epsilon, object strides, int num_threads):
    cdef cnp.int_t[:] c_strides = np.asarray(strides, dtype=np.int)
    cdef int n_filters = output.shape[0]
    cdef int out_height = output.shape[1]
    cdef int out_width = output.shape[2]
    cdef int n_channels = output.shape[3]
    cdef int n_instances = epsilon.shape[0]
    cdef int eps_height = epsilon.shape[1]
    cdef int eps_width = epsilon.shape[2]
    cdef int f, i, j, c, inst, k, l
    np.asarray(output)[:, :, :, :] = 0
    with nogil, parallel(num_threads=num_threads):
        for f in prange(n_filters):
            for inst in prange(n_instances):
                for i in range(out_height):
                    for j in range(out_width):
                        for k in range(eps_height):
                            for l in range(eps_width):
                                for c in range(n_channels):
                                    output[f, i, j, c] += epsilon[inst, k, l, f] * X[inst, k*c_strides[0] + i, l*c_strides[1] + j, c]


def conv_2d_forward(data_t[:, :, :, :] output, data_t[:, :, :, :] X, data_t[:, :, :, :] filters, data_t[:] b, object strides, bint add_bias, int num_threads):
    cdef int a, c, f, i, j, k, l
    cdef cnp.int_t[:] c_strides = np.asarray(strides, dtype=np.int)
    cdef int n_instances = X.shape[0]
    cdef int height = X.shape[1]
    cdef int width = X.shape[2]
    cdef int n_channels = X.shape[3]
    cdef int n_filters = filters.shape[0]
    cdef int filter_height = filters.shape[1]
    cdef int filter_width = filters.shape[2]
    cdef int out_height = output.shape[1]
    cdef int out_width = output.shape[2]
    np.asarray(output)[:, :, :, :] = 0
    with nogil, parallel(num_threads=num_threads):
        for a in prange(n_instances):
            for f in prange(n_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        for c in prange(n_channels):
                            for k in range(filter_height):
                                for l in range(filter_width):
                                    output[a, i, j, f] += filters[f, k, l, c] * X[a, k+i*c_strides[0], l+j*c_strides[1], c]
                        if add_bias:
                            output[a, i, j, f] += b[f]  # Add intercept


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
