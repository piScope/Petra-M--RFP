"""ctypes bridge from the native Maxwellian hot-plasma extension into Numba."""
from ctypes import CFUNCTYPE, c_double, c_int32, c_void_p

import numpy as np
from numba import njit, types
from numba.extending import overload

from . import _rf_dispersion_lkplasma_ext as _native

_hot_std = CFUNCTYPE(
    c_int32, c_double, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_double, c_double, c_double, c_double, c_int32, c_void_p, c_int32,
    c_int32, c_void_p, c_int32, c_void_p)(_native._hot_std_address())


@njit
def _epsilonr_pl_hot_std(w, B, temperatures, denses, masses, charges, Te,
                        ne, npara, nperp, nhrms, terms, use_eye3, nucol):
    """Evaluate the C kernel from Numba with safe contiguous array buffers.

    Specializations compile on demand. Normalize dtype and layout explicitly:
    the C ABI receives pointers without dtype or stride information.
    """
    if B.ndim != 1 or B.size != 3:
        raise ValueError("B must contain three components")
    if (temperatures.ndim != 1 or denses.ndim != 1 or masses.ndim != 1
            or charges.ndim != 1 or nucol.ndim != 1):
        raise ValueError("species and collision arrays must be one-dimensional")
    n = denses.size
    if temperatures.size != n or masses.size != n or charges.size != n:
        raise ValueError("species array lengths must match")
    if nucol.size < n + 1:
        raise ValueError("nucol must contain electron and ion frequencies")
    # Padded shape indexing below also type-checks for invalid array ranks.
    if terms.ndim != 2:
        raise ValueError("terms must be a two-dimensional array")
    if (terms.shape + (0, 0))[0] < n + 1 or (terms.shape + (0, 0))[1] != 8:
        raise ValueError("terms must have one row per species and eight columns")
    B = np.ascontiguousarray(B.astype(np.float64))
    temperatures = np.ascontiguousarray(temperatures.astype(np.float64))
    denses = np.ascontiguousarray(denses.astype(np.float64))
    masses = np.ascontiguousarray(masses.astype(np.float64))
    charges = np.ascontiguousarray(charges.astype(np.int32))
    terms = np.ascontiguousarray(terms.astype(np.int32))
    nucol = np.ascontiguousarray(nucol.astype(np.float64))

    raw = np.empty(18, dtype=np.float64)
    status = _hot_std(w, B.ctypes.data, temperatures.ctypes.data,
                      denses.ctypes.data, masses.ctypes.data, charges.ctypes.data,
                      Te, ne, npara, nperp, np.int32(nhrms), terms.ctypes.data,
                      np.int32((terms.shape + (0, 0))[0]), np.int32(use_eye3), nucol.ctypes.data,
                      np.int32(denses.size), raw.ctypes.data)
    if status != 0:
        raise ValueError("native hot-plasma kernel failed")
    return raw.view(np.complex128).reshape((3, 3))


def call_epsilonr_pl_hot_std(w, B, temps, denses, masses, charges, Te, ne,
                            npara, nperp, nhrms, terms, use_eye3, nucol=None):
    """Overload dispatch used by the public Numba-callable interface."""
    raise NotImplementedError("Use epsilonr_pl_hot_std")


@overload(call_epsilonr_pl_hot_std, strict=False)
def jit_epsilonr_pl_hot_std(w, B, temps, denses, masses, charges, Te, ne,
                           npara, nperp, nhrms, terms, use_eye3, nucol=None):
    if isinstance(nucol, (types.Omitted, types.NoneType)) or nucol is None:
        def without_collisions(w, B, temps, denses, masses, charges, Te, ne,
                               npara, nperp, nhrms, terms, use_eye3, nucol=None):
            # One frequency per species, including electrons.
            collisions = np.zeros(denses.size + 1, dtype=np.float64)
            return _epsilonr_pl_hot_std(
                w, B, temps, denses, masses, charges, Te, ne,
                npara, nperp, nhrms, terms, use_eye3, collisions)
        return without_collisions

    def with_collisions(w, B, temps, denses, masses, charges, Te, ne,
                        npara, nperp, nhrms, terms, use_eye3, nucol=None):
        return _epsilonr_pl_hot_std(
            w, B, temps, denses, masses, charges, Te, ne,
            npara, nperp, nhrms, terms, use_eye3, nucol)
    return with_collisions


@njit
def epsilonr_pl_hot_std(w, B, temps, denses, masses, charges, Te, ne,
                       npara, nperp, nhrms, terms, use_eye3, nucol=None):
    """Return the hot tensor; omitted/None nucol means zero collisions."""
    return call_epsilonr_pl_hot_std(
        w, B, temps, denses, masses, charges, Te, ne,
        npara, nperp, nhrms, terms, use_eye3, nucol)


_hot_rotate = CFUNCTYPE(None, c_void_p, c_void_p, c_void_p, c_void_p)(
    _native._hot_rotate_address())


@njit
def rotate_dielectric(B, K, M):
    """Rotate the hot tensor using the reference B/K coordinate convention."""
    if B.ndim != 1 or B.size != 3 or K.ndim != 1 or K.size != 3:
        raise ValueError("B and K must be three-component vectors")
    if M.shape != (3, 3):
        raise ValueError("M must be a 3-by-3 tensor")
    B = np.ascontiguousarray(B.astype(np.float64))
    K = np.ascontiguousarray(K.astype(np.float64))
    M = np.ascontiguousarray(M.astype(np.complex128))
    raw = np.empty(18, dtype=np.float64)
    _hot_rotate(B.ctypes.data, K.ctypes.data, M.ctypes.data, raw.ctypes.data)
    return raw.view(np.complex128).reshape((3, 3))
