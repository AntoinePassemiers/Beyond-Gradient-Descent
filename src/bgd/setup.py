# -*- coding: utf-8 -*-

import os

import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup

DELETE_GENERATED_C_FILES = False
source_folder = "bgd"
sub_packages = []
source_files = [
    (["operators.c"], "operators"),
]

compile_args = [
    '-fopenmp',
    '-O3',
    '-march=native',
    '-msse',
    '-msse2',
    '-mfma',
    '-mfpmath=sse',
]

libraries = ["m"] if os.name == "posix" else list()
include_dirs = [np.get_include()]

config = Configuration(source_folder, "", "")
for sub_package in sub_packages:
    config.add_subpackage(sub_package)
for sources, extension_name in source_files:
    sources = [os.path.join(source_folder, source) for source in sources]
    extension_name = os.path.splitext(extension_name)[0]
    print(extension_name, sources)
    config.add_extension(
        extension_name, 
        sources=sources,
        include_dirs =include_dirs+[os.curdir],
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_link_args=['-fopenmp']
    )

np_setup(**config.todict())

if DELETE_GENERATED_C_FILES:
    for source_file in source_files:
        filepath = os.path.join(source_folder, source_file[0][0])
        if os.path.isfile(filepath) and os.path.splitext(filepath)[1] == '.c':
            os.remove(filepath)
