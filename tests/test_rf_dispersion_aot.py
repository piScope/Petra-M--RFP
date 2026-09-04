"""Regression tests for the Numba AOT RF dielectric-tensor extensions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from petram.phys.common import rf_dispersion_coldplasma_numba as cold_jit
from petram.phys.common import rf_dispersion_lkplasma_numba as hot_jit
from petram.phys.common.build_rf_dispersion_aot import build_all


def _load_extension(build_dir: Path, module_name: str):
    """Load a pycc extension from the temporary test build directory."""
    extension = next(build_dir.glob(f"{module_name}*.so"))
    spec = importlib.util.spec_from_file_location(module_name, extension)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def aot_modules(tmp_path_factory):
    """Build once per test session; compilation is intentionally exercised."""
    build_dir = tmp_path_factory.mktemp("rf-dispersion-aot")
    build_all(build_dir)
    return (
        _load_extension(build_dir, "_rf_dispersion_coldplasma_aot"),
        _load_extension(build_dir, "_rf_dispersion_lkplasma_aot"),
    )


@pytest.fixture
def plasma_inputs():
    """A small, representative electron-plus-two-ion plasma."""
    return {
        "w": 2.0 * np.pi * 30.0e6,
        "B": np.ascontiguousarray([0.1, 0.2, 0.5], dtype=np.float64),
        "denses": np.ascontiguousarray([5.0e19, 5.0e18], dtype=np.float64),
        "masses": np.ascontiguousarray([3.345e-27, 1.673e-27], dtype=np.float64),
        "charges": np.ascontiguousarray([1, 1], dtype=np.int32),
        "temperatures": np.ascontiguousarray([1.0e4, 1.5e4, 1.5e4], dtype=np.float64),
        "electron_density": 5.0e19,
    }


@pytest.mark.aot
def test_coldplasma_aot_matches_jit(aot_modules, plasma_inputs):
    """The cold-plasma AOT tensor matches the runtime-compiled tensor."""
    cold_aot, _ = aot_modules
    args = (
        plasma_inputs["w"], plasma_inputs["B"], plasma_inputs["denses"],
        plasma_inputs["masses"], plasma_inputs["charges"],
        plasma_inputs["temperatures"], plasma_inputs["electron_density"],
        np.int32(2),  # species-specific Coulomb collision frequencies
    )

    jit_result = cold_jit._epsilonr_pl_cold_std(*args)
    aot_result = cold_aot.epsilonr_pl_cold_std(*args)

    np.testing.assert_allclose(aot_result, jit_result, rtol=1e-12, atol=1e-12)


@pytest.mark.aot
def test_hotplasma_aot_matches_jit(aot_modules, plasma_inputs):
    """The Maxwellian hot-plasma AOT tensor matches its JIT equivalent."""
    _, hot_aot = aot_modules
    terms = np.ones((3, 8), dtype=np.int32)
    nucol = np.ascontiguousarray([0.0, 1.0, 2.0], dtype=np.float64)
    args = (
        plasma_inputs["w"], plasma_inputs["B"], plasma_inputs["temperatures"][1:],
        plasma_inputs["denses"], plasma_inputs["masses"], plasma_inputs["charges"],
        plasma_inputs["temperatures"][0], plasma_inputs["electron_density"],
        10.0, 40.0, np.int32(2), terms, np.int32(1), nucol,
    )

    jit_result = hot_jit._epsilonr_pl_hot_std(*args)
    aot_result = hot_aot.epsilonr_pl_hot_std(*args)

    np.testing.assert_allclose(aot_result, jit_result, rtol=1e-12, atol=1e-12)
