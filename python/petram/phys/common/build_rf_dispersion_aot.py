"""Build Numba AOT extensions for the RF dielectric tensor kernels.

``pip install .`` builds these extensions automatically.  To rebuild them
manually from an installed checkout (or from the source tree), run::

    python -m petram.phys.common.build_rf_dispersion_aot

The command writes ``_rf_dispersion_coldplasma_aot`` and
``_rf_dispersion_lkplasma_aot`` next to this file.  They are deliberately
companion modules: replacing the JIT modules would remove their Python
dispatch/overload interfaces and prevent their functions being used from
other ``@njit`` functions.

The resulting binary is specific to the Python, NumPy, Numba, platform, and
CPU/build environment that created it.  Rebuild it after changing any of
those, or after changing either source kernel module.
"""

from pathlib import Path

from numba.pycc import CC

from petram.phys.common import rf_dispersion_coldplasma_numba as cold
from petram.phys.common import rf_dispersion_lkplasma_numba as hot


_HERE = Path(__file__).resolve().parent


def _compiler(module_name: str, output_dir: Path) -> CC:
    cc = CC(module_name)
    cc.output_dir = str(output_dir)
    return cc


def build_coldplasma(output_dir: Path | None = None) -> None:
    """Compile the typed cold-plasma entry points."""
    cc = _compiler("_rf_dispersion_coldplasma_aot", output_dir or _HERE)

    @cc.export("epsilonr_pl_cold_std", "complex128[:, ::1](float64, float64[::1], float64[::1], float64[::1], int32[::1], float64[::1], float64, int32)")
    def epsilonr_pl_cold_std(w, B, denses, masses, charges, temperatures, electron_density, collision_model):
        return cold._epsilonr_pl_cold_std(w, B, denses, masses, charges, temperatures,
                                          electron_density, collision_model)

    @cc.export("epsilonr_pl_cold_std_scalar_temperature", "complex128[:, ::1](float64, float64[::1], float64[::1], float64[::1], int32[::1], float64, float64, int32)")
    def epsilonr_pl_cold_std_scalar_temperature(w, B, denses, masses, charges, temperature, electron_density, collision_model):
        temperatures = cold.np.zeros(len(masses) + 1) + temperature
        return cold._epsilonr_pl_cold_std(w, B, denses, masses, charges, temperatures,
                                          electron_density, collision_model)

    @cc.export("epsilonr_pl_cold_g", "complex128[:, ::1](float64, float64[::1], float64[::1], float64[::1], int32[::1], float64[::1], float64, int32[:, ::1], int32, int32)")
    def epsilonr_pl_cold_g(w, B, denses, masses, charges, temperatures, electron_density, terms, use_eye3, collision_model):
        return cold._epsilonr_pl_cold_g(w, B, denses, masses, charges, temperatures,
                                        electron_density, terms, use_eye3, collision_model)

    @cc.export("epsilonr_pl_cold_g_scalar_temperature", "complex128[:, ::1](float64, float64[::1], float64[::1], float64[::1], int32[::1], float64, float64, int32[:, ::1], int32, int32)")
    def epsilonr_pl_cold_g_scalar_temperature(w, B, denses, masses, charges, temperature, electron_density, terms, use_eye3, collision_model):
        temperatures = cold.np.zeros(len(masses) + 1) + temperature
        return cold._epsilonr_pl_cold_g(w, B, denses, masses, charges, temperatures,
                                        electron_density, terms, use_eye3, collision_model)

    @cc.export("epsilonr_pl_cold", "complex128[:, :](float64, float64[::1], float64[::1], float64[::1], int32[::1], float64[::1], float64, int32)")
    def epsilonr_pl_cold(w, B, denses, masses, charges, temperatures, electron_density, collision_model):
        return cold._epsilonr_pl_cold(w, B, denses, masses, charges, temperatures,
                                      electron_density, collision_model)

    @cc.export("epsilonr_pl_cold_scalar_temperature", "complex128[:, :](float64, float64[::1], float64[::1], float64[::1], int32[::1], float64, float64, int32)")
    def epsilonr_pl_cold_scalar_temperature(w, B, denses, masses, charges, temperature, electron_density, collision_model):
        temperatures = cold.np.zeros(len(masses) + 1) + temperature
        return cold._epsilonr_pl_cold(w, B, denses, masses, charges, temperatures,
                                      electron_density, collision_model)

    @cc.export("epsilonr_pl_cold_generic", "complex128[:, :](float64, float64[::1], float64[::1], float64[::1], int32[::1], float64[::1], float64, int32[:, ::1], int32, int32)")
    def epsilonr_pl_cold_generic(w, B, denses, masses, charges, temperatures, electron_density, terms, use_eye3, collision_model):
        return cold._epsilonr_pl_cold_generic(w, B, denses, masses, charges, temperatures,
                                              electron_density, terms, use_eye3, collision_model)

    @cc.export("epsilonr_pl_cold_generic_scalar_temperature", "complex128[:, :](float64, float64[::1], float64[::1], float64[::1], int32[::1], float64, float64, int32[:, ::1], int32, int32)")
    def epsilonr_pl_cold_generic_scalar_temperature(w, B, denses, masses, charges, temperature, electron_density, terms, use_eye3, collision_model):
        temperatures = cold.np.zeros(len(masses) + 1) + temperature
        return cold._epsilonr_pl_cold_generic(w, B, denses, masses, charges, temperatures,
                                              electron_density, terms, use_eye3, collision_model)

    cc.compile()


def build_lkplasma(output_dir: Path | None = None) -> None:
    """Compile the typed Maxwellian hot-plasma entry points."""
    cc = _compiler("_rf_dispersion_lkplasma_aot", output_dir or _HERE)

    @cc.export("epsilonr_pl_hot_std", "complex128[:, ::1](float64, float64[::1], float64[::1], float64[::1], float64[::1], int32[::1], float64, float64, float64, float64, int32, int32[:, ::1], int32, float64[::1])")
    def epsilonr_pl_hot_std(w, B, temperatures, denses, masses, charges, electron_temperature, electron_density, npara, nperp, nhrms, terms, use_eye3, nucol):
        return hot._epsilonr_pl_hot_std(w, B, temperatures, denses, masses, charges,
                                        electron_temperature, electron_density, npara, nperp,
                                        nhrms, terms, use_eye3, nucol)

    @cc.export("epsilonr_pl_hot_std_no_nucol", "complex128[:, ::1](float64, float64[::1], float64[::1], float64[::1], float64[::1], int32[::1], float64, float64, float64, float64, int32, int32[:, ::1], int32)")
    def epsilonr_pl_hot_std_no_nucol(w, B, temperatures, denses, masses, charges, electron_temperature, electron_density, npara, nperp, nhrms, terms, use_eye3):
        nucol = hot.np.zeros(len(masses) + 1)
        return hot._epsilonr_pl_hot_std(w, B, temperatures, denses, masses, charges,
                                        electron_temperature, electron_density, npara, nperp,
                                        nhrms, terms, use_eye3, nucol)

    cc.compile()


def build_all(output_dir: Path | None = None) -> None:
    """Build both extensions into *output_dir* (or this package directory)."""
    build_coldplasma(output_dir)
    build_lkplasma(output_dir)


def main() -> None:
    build_all()


if __name__ == "__main__":
    main()
