from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "pf_cython_paralelo",
        sources=["pf_cython_paralelo.pyx"],
        include_dirs=[np.get_include()],  # <--- Esto es lo importante
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="pf_cython_paralelo",
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
)
