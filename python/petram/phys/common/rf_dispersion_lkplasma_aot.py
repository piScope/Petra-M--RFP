"""Convenience interface for the fixed-signature hot-plasma AOT extension."""

import numpy as np

try:
    from . import _rf_dispersion_lkplasma_aot as _aot
except ImportError as error:  # pragma: no cover - depends on local installation
    raise ImportError(
        "The hot-plasma AOT extension is not installed. Run `python -m pip install .` "
        "or `python -m petram.phys.common.build_rf_dispersion_aot`."
    ) from error


def _vector(value, dtype):
    return np.ascontiguousarray(value, dtype=dtype)


def _matrix(value, dtype):
    return np.ascontiguousarray(value, dtype=dtype)


def epsilonr_pl_hot_std(w, B, temps, denses, masses, charges, Te, ne, npara,
                        nperp, nhrms, terms, use_eye3, nucol=None):
    """Return the hot tensor; ``nucol`` may be omitted or supplied as an array."""
    args = (float(w), _vector(B, np.float64), _vector(temps, np.float64),
            _vector(denses, np.float64), _vector(masses, np.float64),
            _vector(charges, np.int32), float(Te), float(ne), float(npara),
            float(nperp), np.int32(nhrms), _matrix(terms, np.int32),
            np.int32(use_eye3))
    if nucol is None:
        return _aot.epsilonr_pl_hot_std_no_nucol(*args)
    return _aot.epsilonr_pl_hot_std(*args, _vector(nucol, np.float64))

