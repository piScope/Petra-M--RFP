## Petra-M(RFP)

Additional module for waves in plasmas

### Install
```bash
python -m pip install .
```

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
The dielectric kernel for cold and lkplasma model is provided by C extensions
or by Numba compiled functions. By default, C extensions are used, and selected
models are loaded to `petram.phys.common.coldplasma` and `petram.phys.common.lkplasma`.

To use the original Numba kernels, set the enviromental variable, before loading the
module. Both facades expose `BACKEND` (`"ext"` or `"numba"`).

```bash
export PETRAM_RFP_USE_NUMBA_DIELECTRIC=1
```
