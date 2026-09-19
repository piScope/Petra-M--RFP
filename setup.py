from setuptools import Extension, setup
from codecs import open
import os
import numpy


def long_description():
    rootdir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(rootdir, 'README.md'), encoding='utf-8') as f:
        return f.read()


def run_setup():
    setup(
        long_description=long_description(),
        long_description_content_type="text/markdown",
        # ctypes obtains kernel addresses from Python methods; only the
        # PyMODINIT_FUNC initialization symbols need dynamic visibility.
        ext_modules=[
            Extension("petram.ext.rfp._rf_dispersion_coldplasma_ext",
                      ["ext/rf_dispersion_coldplasma_ext.c"],
                      include_dirs=[numpy.get_include()],
                      extra_compile_args=["-std=c99", "-fvisibility=hidden"]),
            Extension("petram.ext.rfp._rf_dispersion_lkplasma_ext",
                      ["ext/rf_dispersion_lkplasma_ext.c",
                       "ext/bessel_ive.c"],
                      include_dirs=[numpy.get_include()],
                      extra_compile_args=["-std=c99", "-fvisibility=hidden"]),
        ],)


def main():
    run_setup()


if __name__ == '__main__':
    main()
