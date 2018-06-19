from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

sourcefiles = ['cython_pyx_fde_sc_c2.pyx','cython_fde_sc_c2.c']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cython_fde_sc_c2", sourcefiles, include_dirs=[np.get_include()])],
)
