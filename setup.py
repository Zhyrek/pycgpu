import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# _core cimports pycalphad's PhaseRecord and FastFunction declarations (pycalphad/core/phase_rec.pxd, shipped
# with pycalphad), so it is compiled against the installed pycalphad
extensions = [Extension('pycalphad_cg._core', sources=['pycalphad_cg/_core.pyx'], language='c++',
                        include_dirs=[numpy.get_include()])]
setup(ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}))
