"""ctypes bridge from the native Maxwellian hot-plasma extension into Numba."""
from ctypes import CFUNCTYPE, c_double, c_int32, c_void_p

import numpy as np
from numba import complex128, float64, int32, njit

from . import _rf_dispersion_lkplasma_ext as _native

_hot_std = CFUNCTYPE(
    c_int32, c_double, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_double, c_double, c_double, c_double, c_int32, c_void_p, c_int32,
    c_int32, c_void_p, c_int32, c_void_p)(_native._hot_std_address())


@njit(complex128[:, ::1](
    float64,
    float64[:], float64[:], float64[:], float64[:], int32[:],
    float64, float64, float64, float64, int32, int32[:, :], int32,
    float64[:],
))
def epsilonr_pl_hot_std(w, B, temperatures, denses, masses, charges, Te,
                        ne, npara, nperp, nhrms, terms, use_eye3, nucol):
    """Evaluate the C kernel from Numba with safe contiguous array buffers.

    The explicit signature accepts strided arrays with the ABI's required
    dtypes.  Normalizing here is essential because ctypes passes data pointers
    only; the C implementation has no stride information.
    """
    B = np.ascontiguousarray(B)
    temperatures = np.ascontiguousarray(temperatures)
    denses = np.ascontiguousarray(denses)
    masses = np.ascontiguousarray(masses)
    charges = np.ascontiguousarray(charges)
    terms = np.ascontiguousarray(terms)
    nucol = np.ascontiguousarray(nucol)

    raw = np.empty(18, dtype=np.float64)
    status = _hot_std(w, B.ctypes.data, temperatures.ctypes.data,
                      denses.ctypes.data, masses.ctypes.data, charges.ctypes.data,
                      Te, ne, npara, nperp, nhrms, terms.ctypes.data,
                      np.int32(terms.shape[0]), use_eye3, nucol.ctypes.data,
                      np.int32(len(denses)), raw.ctypes.data)
    if status != 0:
        raise ValueError("native hot-plasma kernel failed")
    return raw.view(np.complex128).reshape((3, 3))
