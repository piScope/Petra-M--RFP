"""Numba-callable cold dielectric interfaces backed by native C kernels."""
from ctypes import CFUNCTYPE, c_double, c_int32, c_void_p

import numpy as np
from numba import njit, types
from numba.extending import overload

from . import _rf_dispersion_coldplasma_ext as _native

__all__ = ["epsilonr_pl_cold_std", "epsilonr_pl_cold_g",
           "epsilonr_pl_cold", "epsilonr_pl_cold_generic", "f_collisions"]

# Keep the extension and ctypes objects alive for the lifetime of compiled callers.
_cold_std = CFUNCTYPE(
    c_int32, c_double, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_int32, c_double, c_int32, c_void_p)(_native._cold_std_address())
_cold_g = CFUNCTYPE(
    c_int32, c_double, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_int32, c_double, c_void_p, c_int32, c_int32, c_int32,
    c_void_p)(_native._cold_g_address())

_cold_rotate = CFUNCTYPE(None, c_void_p, c_void_p, c_void_p)(
    _native._cold_rotate_address())


@njit
def rotate_dielectric(B, M):
    """Rotate a complex 3-by-3 tensor using the native C implementation."""
    if B.ndim != 1 or B.size != 3:
        raise ValueError("B must contain three components")
    if M.shape != (3, 3):
        raise ValueError("M must be a 3-by-3 tensor")
    B = np.ascontiguousarray(B.astype(np.float64))
    M = np.ascontiguousarray(M.astype(np.complex128))
    raw = np.empty(18, dtype=np.float64)
    _cold_rotate(B.ctypes.data, M.ctypes.data, raw.ctypes.data)
    return raw.view(np.complex128).reshape((3, 3))


def _temperatures(Te, n):
    """Normalize the reference interface's scalar or array temperature input."""


@overload(_temperatures)
def _temperature_overload(Te, n):
    if isinstance(Te, types.Array):
        def impl(Te, n):
            return np.ascontiguousarray(Te.astype(np.float64))
        return impl
    if isinstance(Te, (types.Integer, types.Float)):
        def impl(Te, n):
            return np.full(n + 1, Te, dtype=np.float64)
        return impl


@njit
def _validate_buffers(B, denses, masses, charges, temperatures, col_model):
    n = denses.size
    if B.ndim != 1 or B.size != 3:
        raise ValueError("B must contain three components")
    if denses.ndim != 1 or masses.ndim != 1 or charges.ndim != 1:
        raise ValueError("species arrays must be one-dimensional")
    if masses.size != n or charges.size != n:
        raise ValueError("species array lengths must match")
    required = n + 1 if col_model >= 3 else 1
    if temperatures.ndim != 1 or temperatures.size < required:
        raise ValueError("temperature/collision array is too short or not a vector")


@njit
def _buffers(B, denses, masses, charges, Te, col_model):
    temperatures = _temperatures(Te, denses.size)
    _validate_buffers(B, denses, masses, charges, temperatures, col_model)
    return (np.ascontiguousarray(B.astype(np.float64)),
            np.ascontiguousarray(denses.astype(np.float64)),
            np.ascontiguousarray(masses.astype(np.float64)),
            np.ascontiguousarray(charges.astype(np.int32)), temperatures)


@njit
def epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, col_model):
    """Return the unrotated cold tensor; Te may be scalar or an array."""
    B, denses, masses, charges, temperatures = _buffers(
        B, denses, masses, charges, Te, col_model)
    raw = np.empty(18, dtype=np.float64)
    status = _cold_std(w, B.ctypes.data, denses.ctypes.data, masses.ctypes.data,
                       charges.ctypes.data, temperatures.ctypes.data,
                       np.int32(denses.size), ne, np.int32(col_model), raw.ctypes.data)
    if status != 0:
        raise ValueError("native cold-plasma kernel failed")
    return raw.view(np.complex128).reshape((3, 3))


@njit
def epsilonr_pl_cold_g(w, B, denses, masses, charges, Te, ne, terms,
                      use_eye3, col_model):
    """Return the unrotated tensor with species-specific contribution flags."""
    B, denses, masses, charges, temperatures = _buffers(
        B, denses, masses, charges, Te, col_model)
    # Padded shape indexing below also type-checks for invalid array ranks.
    if terms.ndim != 2:
        raise ValueError("terms must be a two-dimensional array")
    if (terms.shape + (0, 0))[0] < denses.size + 1 or (terms.shape + (0, 0))[1] != 5:
        raise ValueError("terms must have one row per species and five columns")
    terms = np.ascontiguousarray(terms.astype(np.int32))
    raw = np.empty(18, dtype=np.float64)
    status = _cold_g(w, B.ctypes.data, denses.ctypes.data, masses.ctypes.data,
                     charges.ctypes.data, temperatures.ctypes.data,
                     np.int32(denses.size), ne, terms.ctypes.data,
                     np.int32((terms.shape + (0, 0))[0]), np.int32(use_eye3),
                     np.int32(col_model), raw.ctypes.data)
    if status != 0:
        raise ValueError("native generalized cold-plasma kernel failed")
    return raw.view(np.complex128).reshape((3, 3))


@njit
def epsilonr_pl_cold(w, B, denses, masses, charges, Te, ne, col_model):
    """Return the standard tensor aligned with the magnetic field B."""
    return rotate_dielectric(B, epsilonr_pl_cold_std(
        w, B, denses, masses, charges, Te, ne, col_model))


@njit
def epsilonr_pl_cold_generic(w, B, denses, masses, charges, Te, ne, terms,
                            use_eye3, col_model):
    """Return the contribution-selected tensor aligned with B."""
    return rotate_dielectric(B, epsilonr_pl_cold_g(
        w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model))


_collisions = CFUNCTYPE(None, c_void_p, c_void_p, c_void_p, c_int32,
                       c_double, c_double, c_void_p)(_native._collisions_address())


@njit
def f_collisions(denses, masses, charges, Te, ne):
    """Return electron and ion collision frequencies using the native kernel."""
    if denses.ndim != 1 or masses.ndim != 1 or charges.ndim != 1:
        raise ValueError("species arrays must be one-dimensional")
    if masses.size != denses.size or charges.size != denses.size:
        raise ValueError("species array lengths must match")
    denses = np.ascontiguousarray(denses.astype(np.float64))
    masses = np.ascontiguousarray(masses.astype(np.float64))
    charges = np.ascontiguousarray(charges.astype(np.int32))
    frequencies = np.empty(denses.size + 1, dtype=np.float64)
    _collisions(denses.ctypes.data, masses.ctypes.data, charges.ctypes.data,
                np.int32(denses.size), Te, ne, frequencies.ctypes.data)
    return frequencies
