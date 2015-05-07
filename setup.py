from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("basis1d.c1dints",["cython/c1dints.pyx"])]


with open('README.md') as file:
	long_description = file.read()

setup(name = 'basis1d',
      version = '0.1',
      description = '1D density represented as a linear combination of gaussian type basis.',
      long_description = long_description,
      author = 'Li Li',
      author_email = 'aiki.nogard@gmail.com',
      install_requires = ['numpy'],
      packages = ['basis1d'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      include_dirs=[numpy.get_include()],
      ) 