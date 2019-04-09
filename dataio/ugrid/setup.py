"""
Build with:
     python setup.py build_ext --inplace

See this site for building on windows-64:
        https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
"""

#from distutils.core import setup
#from distutils.extension import Extension
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import os
import numpy

# RH: on linux, default CC is fine, generally don't have cl.
# os.environ["CC"]='cl'

extensions =[
    Extension("ugridutils",["ugridutils.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],),
    Extension("searchutils",["searchutils.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3','-ffast-math','-march=native','-fopenmp'],
        extra_link_args=['-fopenmp'],),
]

setup(
    name = "Shallow water utilities",
    ext_modules = cythonize(extensions)
)
