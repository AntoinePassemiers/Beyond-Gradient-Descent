# conv.pyx
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


def conv_2d_backward_weights(data_t[:,:,:,:] output, data_t[:,:,:,:] X,
                             data_t[:,:,:,:] epsilon, object strides,
                             object dilations, int num_threads):
    cdef cnp.int_t[:] c_strides = np.asarray(strides, dtype=np.int)
    cdef cnp.int_t[:] c_dilations = np.asarray(dilations, dtype=np.int)
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
                                    output[f, i, j, c] += epsilon[inst, k, l, f] * X[inst, k*c_strides[0] + i*c_dilations[0], l*c_strides[1] + j*c_dilations[1], c]


def conv_2d_forward(data_t[:, :, :, :] output, data_t[:, :, :, :] X,
                    data_t[:, :, :, :] filters, data_t[:] b, object strides,
                    object dilations, bint add_bias, int num_threads):
    cdef int a, c, f, i, j, k, l
    cdef cnp.int_t[:] c_strides = np.asarray(strides, dtype=np.int)
    cdef cnp.int_t[:] c_dilations = np.asarray(dilations, dtype=np.int)
    cdef int n_instances = X.shape[0]
    cdef int height = X.shape[1]
    cdef int width = X.shape[2]
    cdef int n_channels = X.shape[3]
    cdef int n_filters = filters.shape[0]
    cdef int filter_height = filters.shape[1]
    cdef int filter_width = filters.shape[2]
    cdef int out_height = output.shape[1]
    cdef int out_width = output.shape[2]
    if add_bias:
        np.asarray(output)[:, :, :, :] = np.asarray(b[:])
    else:
        np.asarray(output)[:, :, :, :] = 0
    with nogil, parallel(num_threads=num_threads):
        for a in prange(n_instances):
            for f in range(n_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        for c in range(n_channels):
                            for k in range(filter_height):
                                for l in range(filter_width):
                                    output[a, i, j, f] += filters[f, k, l, c] * X[a, i*c_strides[0] + k*c_dilations[0], j*c_strides[1] + l*c_dilations[1], c]


def conv_2d_forward_sse(cnp.float32_t[:, :, :, :] output,
                        cnp.float32_t[:, :, :, :] X,
                        cnp.float32_t[:, :, :, :] filters,
                        cnp.float32_t[:] b,
                        object strides,
                        bint add_bias):

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

    cdef int rc, drc
    cdef cnp.float32_t[8] buf
    cdef __m128 _F
    cdef __m128 _I
    cdef __m128 _O

    np.asarray(output)[:, :, :, :] = 0
    with nogil:
        for a in range(n_instances):
            for f in range(n_filters):
                for i in range(out_height):
                    for j in range(out_width):
                        for k in range(filter_height):
                            for l in range(filter_width):
                                rc = 0
                                while rc < n_channels:

                                    # Compute upper bound to avoid out-of-bound errors
                                    drc = min(4, n_channels-rc)
                                    memset(&buf[0], 0x00, 8)
                                    memcpy(&buf[0], &filters[f, k, l, rc], 4*drc)
                                    memcpy(&buf[4], &X[a, k+i*c_strides[0], l+j*c_strides[1], rc], 4*drc)
                                    rc += drc

                                    # Elemwise multiplication
                                    _F = _mm_loadu_ps(&buf[0])
                                    _I = _mm_loadu_ps(&buf[4])
                                    _O = _mm_mul_ps(_F, _I)

                                    # Sum content of _O and store result in buffer
                                    _O = _mm_hadd_ps(_O, _O)
                                    _O = _mm_hadd_ps(_O, _O)
                                    _mm_store_ps(&buf[0], _O)

                                    output[a, i, j, f] += buf[0]

                        if add_bias:
                            output[a, i, j, f] += b[f]  # Add intercept
