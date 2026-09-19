## Petra-M(RFP)

This module provides additional module for waves in plasmas

### EM3D : Frequency domain Maxwell equation in 3D
  Domain:   
 *    EM3D_ColdPlasma  : Cold magnetised plasma
 *    EM3D_LkPlasma    : local-K approx. plasma
 
### EM2Da : Frequency domain Maxwell equation in 2D axissymetric space
  Domain:   
 *    EM2Da_ColdPlasma  : Cold magnetised plasma
 *    EM2Da_LkPlasma    : local-K approx. plasma

### EM2D : Frequency domain Maxwell equation in 2D space
  Domain:   
 *    EM2D_ColdPlasma  : Cold magnetised plasma
 *    EM2D_LkPlasma    : local-K approx. plasma

### EM1D : Frequency domain Maxwell equation in 1D
  Domain:   
 *    EM1D_ColdPlasma  : Cold magnetised plasma
 *    EM1D_LkPlasma    : local-K approx. plasma 
 

### Dielectric backend

The cold and hot dielectric APIs are available from
`petram.phys.common.coldplasma` and `petram.phys.common.lkplasma`.
The C extensions are selected by default. To use the original Numba kernels:

```bash
export PETRAM_RFP_USE_NUMBA_DIELECTRIC=1
```

Set this before starting the application. Set it to `0` (or unset it) for C.
Both facades expose `BACKEND` (`"ext"` or `"numba"`). Selection is shared and
read once at first import; restart the process to change it. Reloading a facade
does not update existing imported references or compiled Numba callers.
The shared wave-vector helpers remain Numba functions in both modes.
A missing C extension raises an import error rather than silently switching.

`python -m pip install .` builds the C extensions under `petram.ext.rfp` from
sources in `ext/`. Numba AOT compilation is no longer part of installation.

### Simulation regression tests

The examples can be run directly with `python model.py` from a working directory
(using a relative path to the script when working elsewhere). To compare both
dielectric backends on the cold and hot examples:

```bash
python -m pytest tests/test_coldplasma.py tests/test_lkplasma.py -v -s
```

See [tests/README.md](tests/README.md) for dependencies, output comparisons,
measured differences, and tolerance choices.
