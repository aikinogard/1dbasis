from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'i am me',
  ext_modules = cythonize("c1dints.pyx"),
  include_dirs=[numpy.get_include()]
)