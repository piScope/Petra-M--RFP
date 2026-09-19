from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py
from codecs import open
import os
import sys
import numpy


class build_py(_build_py):
    """Add the RF Numba AOT extensions to built wheels."""

    def run(self):
        super().run()
        source_root = os.path.join(os.path.dirname(__file__), "python")
        sys.path.insert(0, source_root)
        try:
            from petram.phys.common.build_rf_dispersion_aot import build_all
            output_dir = os.path.join(self.build_lib, "petram", "phys", "common")
            build_all(output_dir)
        finally:
            sys.path.pop(0)


def long_description():
    rootdir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(rootdir, 'README.md'), encoding='utf-8') as f:
        return f.read()


def run_setup():
    setup(
        long_description=long_description(),
        long_description_content_type="text/markdown",
        cmdclass={"build_py": build_py},
        # ctypes obtains kernel addresses from Python methods; only the
        # PyMODINIT_FUNC initialization symbols need dynamic visibility.
        ext_modules=[
            Extension("petram.phys.common._rf_dispersion_coldplasma_ext",
                      ["python/petram/phys/common/c_ext/rf_dispersion_coldplasma_ext.c"],
                      include_dirs=[numpy.get_include()],
                      extra_compile_args=["-std=c99", "-fvisibility=hidden"]),
            Extension("petram.phys.common._rf_dispersion_lkplasma_ext",
                      ["python/petram/phys/common/c_ext/rf_dispersion_lkplasma_ext.c",
                       "python/petram/phys/common/c_ext/bessel_ive.c"],
                      include_dirs=[numpy.get_include()],
                      extra_compile_args=["-std=c99", "-fvisibility=hidden"]),
        ],)


def main():
    run_setup()


if __name__ == '__main__':
    main()
