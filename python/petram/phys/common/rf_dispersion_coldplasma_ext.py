"""ctypes bridge from the native cold-plasma extension into Numba.

The compiled module is intentionally private: this module owns the ctypes
objects, preventing their function pointers from becoming invalid.
"""
from ctypes import CFUNCTYPE, c_double, c_int32, c_void_p

import numpy as np
from numba import complex128, float64, int32, njit

from . import _rf_dispersion_coldplasma_ext as _native

_cold_std = CFUNCTYPE(c_int32, c_double, c_void_p, c_void_p, c_void_p,
                      c_void_p, c_void_p, c_int32, c_double, c_int32,
                      c_void_p)(
    _native._cold_std_address())


@njit(complex128[:, ::1](
    float64,
    float64[:], float64[:], float64[:], int32[:], float64[:],
    float64, int32,
))
def epsilonr_pl_cold_std(w, B, denses, masses, charges, temperatures,
                         electron_density, collision_model):
    """Evaluate the C kernel from Numba with safe contiguous array buffers.

    The explicit signature accepts strided arrays with the ABI's required
    dtypes.  Normalizing here is essential because ctypes passes data pointers
    only; the C implementation has no stride information.
    """
    B = np.ascontiguousarray(B)
    denses = np.ascontiguousarray(denses)
    masses = np.ascontiguousarray(masses)
    charges = np.ascontiguousarray(charges)
    temperatures = np.ascontiguousarray(temperatures)

    raw = np.empty(18, dtype=np.float64)
    status = _cold_std(w, B.ctypes.data, denses.ctypes.data, masses.ctypes.data,
                       charges.ctypes.data, temperatures.ctypes.data,
                       np.int32(len(denses)), electron_density, collision_model,
                       raw.ctypes.data)
    if status != 0:
        raise ValueError("native cold-plasma kernel failed")
    return raw.view(np.complex128).reshape((3, 3))
