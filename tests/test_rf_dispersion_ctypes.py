"""Regression coverage for the native C ABI and its Numba ctypes bridge."""

import numpy as np
import pytest


def _inputs():
    return (
        2.0 * np.pi * 30.0e6,
        np.ascontiguousarray([0.1, 0.2, 0.5], dtype=np.float64),
        np.ascontiguousarray([5.0e19, 5.0e18], dtype=np.float64),
        np.ascontiguousarray([3.345e-27, 1.673e-27], dtype=np.float64),
        np.ascontiguousarray([1, 1], dtype=np.int32),
        np.ascontiguousarray([1.0e4, 1.5e4, 1.5e4], dtype=np.float64),
        5.0e19,
        np.int32(2),
    )


def test_cold_ctypes_kernel_matches_jit():
    pytest.importorskip("petram.phys.common._rf_dispersion_coldplasma_ext")
    from petram.phys.common import rf_dispersion_coldplasma_ext as native
    from petram.phys.common import rf_dispersion_coldplasma_numba as jit

    args = _inputs()
    expected = jit._epsilonr_pl_cold_std(*args)
    np.testing.assert_allclose(native.epsilonr_pl_cold_std(*args), expected,
                               rtol=1e-14, atol=1e-12)


@pytest.mark.parametrize("value", [
    0.0,
    0.1,
    -3.5,
    9.999,
    10.001,
    4.0 + 2.0j,
    -7.0 + 1.5j,
])
def test_hot_native_zfunc_matches_numba(value):
    """The C rational and asymptotic Z-function branches match Numba."""
    native = pytest.importorskip(
        "petram.phys.common._rf_dispersion_lkplasma_ext")
    from petram.phys.common.numba_zfunc import zfunc

    np.testing.assert_allclose(native._zfunc(value), zfunc(value),
                               rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("order", range(21))
@pytest.mark.parametrize("value", [0.0, 1e-8, 0.1, 1.0, 3.0,
                                    10.0, 30.0, 100.0, 500.0])
def test_hot_native_ive_matches_petram_bessel(order, value):
    """The C real-λ specialization matches the active Bessel implementation."""
    native = pytest.importorskip(
        "petram.phys.common._rf_dispersion_lkplasma_ext")
    from petram.helper.bessel import ive

    np.testing.assert_allclose(native._ive(order, value), ive(order, value),
                               rtol=2e-13, atol=2e-15)
