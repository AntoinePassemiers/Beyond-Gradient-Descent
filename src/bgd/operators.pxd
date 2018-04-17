# -*- coding: utf-8 -*-
# operators.pxd
# author : Antoine Passemiers, Robin Petit
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()


cdef extern from "emmintrin.h": # SSE2
    # Make __128 and __m128d behave like doubles
    #   -> Cython will find the actual definition of __128 and __m128d
    #      in emmintrin.h anyway
    ctypedef double __m128
    ctypedef double __m128d

    __m128  _mm_loadu_ps(float *mem_addr) nogil
    __m128d _mm_loadu_pd(double *mem_addr) nogil
    __m128  _mm_add_ps(__m128 a, __m128 b) nogil
    __m128d _mm_add_pd(__m128d a, __m128d b) nogil
    __m128d _mm_mul_pd(__m128 a, __m128 b) nogil
    __m128d _mm_mul_pd(__m128d a, __m128d __B) nogil
    void    _mm_store_pd(double *mem_addr, __m128d a) nogil
    void    _mm_store_pd(double *mem_addr, __m128d a) nogil
