#from distutils.core import setup
#from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
from setuptools import setup, Extension

import numpy as np

sourcefiles = ['cy_fde/cy_spn.pyx','cy_fde/spn.c']

BLAS_PATH =  "/opt/OpenBLAS/include/"

ext_modules = [
Extension("cy_fde_spn",
sourcefiles,
include_dirs=[np.get_include(), BLAS_PATH] )]



setup(
    packages=["cy_fde"],
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)
