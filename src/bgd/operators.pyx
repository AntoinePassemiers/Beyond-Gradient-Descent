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


cdef fused data_t:
    #cnp.int_t
    #cnp.int8_t
    #cnp.int16_t
    #cnp.int32_t
    #cnp.int64_t
    cnp.float_t
    cnp.float32_t
    cnp.float64_t


def test():
    with nogil:
        pass
    print("c'est loin mais c'est beau !")



# TODO: add padding

### Needs to be debugged after weight update has been corrected
def conv_2d_backward(data_t[:, :, :, :] output, data_t[:, :, :, :] delta, data_t[:, :, :, :] filters, object strides):
    cdef cnp.int_t[:] c_strides = np.asarray(strides, dtype=np.int)
    cdef unsigned int n_instances = delta.shape[0]
    cdef unsigned int height = delta.shape[1]
    cdef unsigned int width = delta.shape[2]
    cdef unsigned int n_channels = delta.shape[3]
    cdef unsigned int filter_height = filters.shape[1]
    cdef unsigned int filter_width = filters.shape[2]
    cdef unsigned int n_filters = filters.shape[3]
    cdef unsigned int out_height = (height + filter_height - 1) // c_strides[0]
    cdef unsigned int out_width = (width + filter_width - 1) // c_strides[1]
    cdef unsigned int instance, i, j, f, h, w, c, alpha, beta, h_0
    np.asarray(output)[:, :, :, :] = 0
    with nogil:
        for instance in range(n_instances):
            for i in range(out_height):
                for j in range(out_width):
                    # from here, index twisting should be continued in order to avoid looping for nothing
                    h_0 = 0 if i <= filter_height else i-filter_height-1
                    for h in range(h_0, min(height, i+1)):
                        # And this, here, is some pretty nasty non-twisting aftermath (after math, get it? I am lonely)
                        if 0 <= i-h < filter_height:
                            w_0 = 0 if j <= filter_width else j-filter_width-1
                            for w in range(w_0, min(width, j+1)):
                                if 0 <= j-w < filter_width:
                                    for c in range(n_channels):
                                        for f in range(n_filters):
                                            output[instance, i, j, f] += delta[instance, h, w, c] * filters[c, i-h, j-w, f]

def conv_2d_backward_weights(data_t[:,:,:,:] output, data_t[:,:,:,:] X, data_t[:,:,:,:] epsilon, object strides):
    cdef cnp.int_t[:] c_strides = np.asarray(strides, dtype=np.int)
    cdef unsigned int n_filters = output.shape[0]
    cdef unsigned int out_height = output.shape[1]
    cdef unsigned int out_width = output.shape[2]
    cdef unsigned int n_channels = output.shape[3]
    cdef unsigned int n_instances = epsilon.shape[0]
    cdef unsigned int eps_height = epsilon.shape[1]
    cdef unsigned int eps_width = epsilon.shape[2]
    cdef unsigned int f, i, j, c, inst, k, l
    with nogil:
        for f in range(n_filters):
            for i in range(out_height):
                for j in range(out_width):
                    for c in range(n_channels):
                        # beta_0
                        for inst in range(n_instances):
                            # beta_1
                            for k in range(eps_height):
                                # beta_2
                                for l in range(eps_width):
                                    output[f, i, j, c] += epsilon[inst, k, l, f] * X[inst, k*c_strides[0] + i, l*c_strides[1] + j, c]
    return np.asarray(output)

def conv_2d_forward(data_t[:, :, :, :] output, data_t[:, :, :, :] X, data_t[:, :, :, :] filters, data_t[:] b, object strides, bint add_bias):
    cdef unsigned int a, c, f, i, j, k, l
    cdef cnp.int_t[:] c_strides = np.asarray(strides, dtype=np.int)
    cdef unsigned int n_instances = X.shape[0]
    cdef unsigned int height = X.shape[1]
    cdef unsigned int width = X.shape[2]
    cdef unsigned int n_channels = X.shape[3]
    cdef unsigned int n_filters = filters.shape[0]
    cdef unsigned int filter_height = filters.shape[1]
    cdef unsigned int filter_width = filters.shape[2]
    cdef unsigned int out_height = output.shape[1]
    cdef unsigned int out_width = output.shape[2]
    np.asarray(output)[:, :, :, :] = 0
    with nogil:
        for a in range(n_instances):
            for i in range(out_height):
                for j in range(out_width):
                    for c in range(n_channels):
                        for f in range(n_filters):
                            for k in range(filter_height):
                                for l in range(filter_width):
                                    output[a, i, j, f] += filters[f, k, l, c] * X[a, k+i*c_strides[0], l+j*c_strides[1], c]
                        if add_bias:
                            output[a, i, j, f] += b[f]  # Add intercept
    return np.asarray(output)[:n_instances, :out_height, :out_width, :n_filters]
