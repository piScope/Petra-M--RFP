"""Stable hot-plasma API with shared wave-vector and frequency helpers."""
from ._dielectric_backend import BACKEND, USE_NUMBA
from .rf_dispersion_lkplasma_helper import (
    eval_npara_nperp, eval_kpe_std, eval_kpe_em1d, eval_kpe_em2d, eval_kpe_em2da,
)
from .rf_plasma_wc_wp import wce, wci, wpesq, wpisq

if USE_NUMBA:
    from .rf_dispersion_lkplasma_numba import epsilonr_pl_hot_std, rotate_dielectric
else:
    from .rf_dispersion_lkplasma_ext import epsilonr_pl_hot_std, rotate_dielectric

__all__ = ["BACKEND", "epsilonr_pl_hot_std", "rotate_dielectric",
           "eval_npara_nperp", "eval_kpe_std", "eval_kpe_em1d", "eval_kpe_em2d",
           "eval_kpe_em2da", "wce", "wci", "wpesq", "wpisq"]
