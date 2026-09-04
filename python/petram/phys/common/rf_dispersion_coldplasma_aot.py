"""Convenience interface for the fixed-signature cold-plasma AOT extension.

Build the companion extension first with ``python -m pip install .`` or
``python -m petram.phys.common.build_rf_dispersion_aot``.
"""

import numpy as np

try:
    from . import _rf_dispersion_coldplasma_aot as _aot
except ImportError as error:  # pragma: no cover - depends on local installation
    raise ImportError(
        "The cold-plasma AOT extension is not installed. Run `python -m pip install .` "
        "or `python -m petram.phys.common.build_rf_dispersion_aot`."
    ) from error


def _vector(value, dtype):
    return np.ascontiguousarray(value, dtype=dtype)


def _matrix(value, dtype):
    return np.ascontiguousarray(value, dtype=dtype)


def _cold_arguments(B, denses, masses, charges):
    return (_vector(B, np.float64), _vector(denses, np.float64),
            _vector(masses, np.float64), _vector(charges, np.int32))


def epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, col_model):
    """Return the unrotated cold tensor; ``Te`` may be a scalar or array."""
    B, denses, masses, charges = _cold_arguments(B, denses, masses, charges)
    args = (float(w), B, denses, masses, charges)
    if np.isscalar(Te):
        return _aot.epsilonr_pl_cold_std_scalar_temperature(
            *args, float(Te), float(ne), np.int32(col_model))
    return _aot.epsilonr_pl_cold_std(
        *args, _vector(Te, np.float64), float(ne), np.int32(col_model))


def epsilonr_pl_cold_g(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
    """Return the generalized unrotated cold tensor; ``Te`` may be scalar/array."""
    B, denses, masses, charges = _cold_arguments(B, denses, masses, charges)
    args = (float(w), B, denses, masses, charges)
    terms = _matrix(terms, np.int32)
    if np.isscalar(Te):
        return _aot.epsilonr_pl_cold_g_scalar_temperature(
            *args, float(Te), float(ne), terms, np.int32(use_eye3), np.int32(col_model))
    return _aot.epsilonr_pl_cold_g(
        *args, _vector(Te, np.float64), float(ne), terms,
        np.int32(use_eye3), np.int32(col_model))


def epsilonr_pl_cold(w, B, denses, masses, charges, Te, ne, col_model):
    """Return the rotated cold tensor; ``Te`` may be a scalar or array."""
    B, denses, masses, charges = _cold_arguments(B, denses, masses, charges)
    args = (float(w), B, denses, masses, charges)
    if np.isscalar(Te):
        return _aot.epsilonr_pl_cold_scalar_temperature(
            *args, float(Te), float(ne), np.int32(col_model))
    return _aot.epsilonr_pl_cold(
        *args, _vector(Te, np.float64), float(ne), np.int32(col_model))


def epsilonr_pl_cold_generic(w, B, denses, masses, charges, Te, ne, terms, use_eye3, col_model):
    """Return the rotated generalized tensor; ``Te`` may be scalar or array."""
    B, denses, masses, charges = _cold_arguments(B, denses, masses, charges)
    args = (float(w), B, denses, masses, charges)
    terms = _matrix(terms, np.int32)
    if np.isscalar(Te):
        return _aot.epsilonr_pl_cold_generic_scalar_temperature(
            *args, float(Te), float(ne), terms, np.int32(use_eye3), np.int32(col_model))
    return _aot.epsilonr_pl_cold_generic(
        *args, _vector(Te, np.float64), float(ne), terms,
        np.int32(use_eye3), np.int32(col_model))

