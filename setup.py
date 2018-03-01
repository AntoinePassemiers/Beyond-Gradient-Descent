# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

import os
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


NAME = 'bgd'
SRC_FOLDER = 'src'
DELETE_GENERATED_C_FILES = True

libraries = ["m"] if os.name == "posix" else list()
include_dirs = [np.get_include()]

extensions = list()
for filename in os.listdir(SRC_FOLDER):
    filepath = os.path.join(SRC_FOLDER, filename)
    head, tail = os.path.splitext(filename)
    if tail == '.pyx':
        extensions.append(Extension(
            NAME + '.' + head,
            sources=[filepath],
            libraries=libraries,
            include_dirs=include_dirs))

setup(
    name = NAME,
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions
)

if DELETE_GENERATED_C_FILES:
    # Delete generated C files
    for filename in os.listdir(SRC_FOLDER):
        filepath = os.path.join(SRC_FOLDER, filename)
        head, tail = os.path.splitext(filename)
        if tail == '.c':
            if os.path.exists(filepath):
                print('Delete generated file %s' % filepath)
                os.remove(filepath)