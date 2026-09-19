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
    pytest.importorskip("petram.ext.rfp._rf_dispersion_coldplasma_ext")
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
        "petram.ext.rfp._rf_dispersion_lkplasma_ext")
    from petram.phys.common.numba_zfunc import zfunc

    np.testing.assert_allclose(native._zfunc(value), zfunc(value),
                               rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("order", range(21))
@pytest.mark.parametrize("value", [0.0, 1e-8, 0.1, 1.0, 3.0,
                                    10.0, 30.0, 100.0, 500.0])
def test_hot_native_ive_matches_petram_bessel(order, value):
    """The C real-λ specialization matches the active Bessel implementation."""
    native = pytest.importorskip(
        "petram.ext.rfp._rf_dispersion_lkplasma_ext")
    from petram.helper.bessel import ive

    np.testing.assert_allclose(native._ive(order, value), ive(order, value),
                               rtol=2e-13, atol=2e-15)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_hot_native_ive_matches_scipy_dense(order):
    """Cover 0 <= x <= 1000, including both sides of branch transitions."""
    native = pytest.importorskip(
        "petram.ext.rfp._rf_dispersion_lkplasma_ext")
    scipy_special = pytest.importorskip("scipy.special")
    boundaries = np.array([2.0 * np.sqrt(order + 1.0), 20.4])
    values = np.unique(np.concatenate([
        np.linspace(0.0, 1000.0, 100001),
        np.geomspace(1e-12, 1000.0, 10001),
        boundaries,
        np.nextafter(boundaries, 0.0),
        np.nextafter(boundaries, np.inf),
    ]))
    actual = np.array([native._ive(order, float(x)) for x in values])
    expected = scipy_special.ive(order, values)
    assert np.isfinite(actual).all()
    # Relative-only tolerance also checks the very small values near x=0.
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0.0)


@pytest.mark.parametrize("hermitian,antihermitian", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_hot_native_term_projection(hermitian, antihermitian):
    """Check term filtering against matrix projections, without a JIT oracle."""
    native = pytest.importorskip(
        "petram.ext.rfp._rf_dispersion_lkplasma_ext")
    w, b, densities, masses, charges, temperatures, ne, _ = _inputs()
    terms = np.ones((3, 8), dtype=np.int32)

    def evaluate():
        return native.epsilonr_pl_hot_std(
            w, b, temperatures[1:], densities, masses, charges,
            temperatures[0], ne, 10.0, 40.0, 20, terms, 0, np.zeros(3))

    full = evaluate()
    adjoint = full.conj().T
    expected = (hermitian * (full + adjoint) / 2.0
                + antihermitian * (full - adjoint) / 2.0)
    # Ensure the antisymmetric entries exercise both parts of the projection.
    assert np.all(np.abs(full[[0, 1], [1, 2]].real) > 0)
    assert np.all(np.abs(full[[0, 1], [1, 2]].imag) > 0)
    terms[:, 6] = hermitian
    terms[:, 7] = antihermitian
    np.testing.assert_allclose(evaluate(), expected, rtol=2e-14, atol=1e-12)
