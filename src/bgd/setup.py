import os

import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup

DELETE_GENERATED_C_FILES = False
SRC_FOLDER = 'bgd'
SUB_PACKAGES = [
    'layers'
]
SRC_FILES = [
    (['layers/conv.c'], 'layers.conv'),
    (['layers/max_pooling.c'], 'layers.max_pooling'),
]

COMPILE_ARGS = [
    '-fopenmp',
    '-O3',
    '-march=native',
    '-msse',
    '-msse2',
    '-mfma',
    '-mfpmath=sse',
]

LIBRARIES = ["m"] if os.name == "posix" else list()
INCLUDE_DIRS = [np.get_include()]

CONFIG = Configuration(SRC_FOLDER, "", "")
for sub_package in SUB_PACKAGES:
    CONFIG.add_subpackage(sub_package)
for sources, extension_name in SRC_FILES:
    sources = [os.path.join(SRC_FOLDER, source) for source in sources]
    #extension_name = os.path.splitext(extension_name)[0]
    print(extension_name, sources)
    CONFIG.add_extension(
        extension_name,
        sources=sources,
        include_dirs=INCLUDE_DIRS + [os.curdir],
        libraries=LIBRARIES,
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=['-fopenmp']
    )

np_setup(**CONFIG.todict())

if DELETE_GENERATED_C_FILES:
    for source_file in SRC_FILES:
        filepath = os.path.join(SRC_FOLDER, source_file[0][0])
        if os.path.isfile(filepath) and os.path.splitext(filepath)[1] == '.c':
            os.remove(filepath)
