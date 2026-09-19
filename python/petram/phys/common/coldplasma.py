"""Stable cold-plasma API; select backend before importing this module."""
from ._dielectric_backend import BACKEND, USE_NUMBA

if USE_NUMBA:
    from .rf_dispersion_coldplasma_numba import (
        epsilonr_pl_cold_std, epsilonr_pl_cold_g, epsilonr_pl_cold,
        epsilonr_pl_cold_generic, f_collisions, rotate_dielectric,
    )
else:
    from .rf_dispersion_coldplasma_ext import (
        epsilonr_pl_cold_std, epsilonr_pl_cold_g, epsilonr_pl_cold,
        epsilonr_pl_cold_generic, f_collisions, rotate_dielectric,
    )

__all__ = ["BACKEND", "epsilonr_pl_cold_std", "epsilonr_pl_cold_g",
           "epsilonr_pl_cold", "epsilonr_pl_cold_generic", "f_collisions",
           "rotate_dielectric"]
