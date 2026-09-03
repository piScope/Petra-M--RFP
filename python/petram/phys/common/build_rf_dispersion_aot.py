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

from __future__ import annotations

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

    @cc.export("f_collisions", "float64[:](float64[::1], float64[::1], int32[::1], float64, float64)")
    def f_collisions(denses, masses, charges, temperature, electron_density):
        return cold.f_collisions(denses, masses, charges, temperature, electron_density)

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

    @cc.export("epsilonr_pl_cold_generic", "complex128[:, :](float64, float64[::1], float64[::1], float64[::1], int32[::1], float64[::1], float64, int32[:, ::1], int32, int32)")
    def epsilonr_pl_cold_generic(w, B, denses, masses, charges, temperatures, electron_density, terms, use_eye3, collision_model):
        return cold._epsilonr_pl_cold_generic(w, B, denses, masses, charges, temperatures,
                                              electron_density, terms, use_eye3, collision_model)

    @cc.export("rotate_dielectric", "complex128[:, :](float64[::1], complex128[:, ::1])")
    def rotate_dielectric(B, dielectric):
        return cold.rotate_dielectric(B, dielectric)

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

    @cc.export("rotate_dielectric", "complex128[:, :](float64[::1], float64[::1], complex128[:, ::1])")
    def rotate_dielectric(B, K, dielectric):
        return hot.rotate_dielectric(B, K, dielectric)

    @cc.export("eval_npara_nperp", "complex128[:](float64[::1], float64, float64[::1], int64, complex128[:, ::1])")
    def eval_npara_nperp(point, omega, kpakpe, mode, cold_dielectric):
        return hot.eval_npara_nperp(point, omega, kpakpe, mode, cold_dielectric)

    for name, function in (("eval_kpe_std", hot.eval_kpe_std),
                           ("eval_kpe_em1d", hot.eval_kpe_em1d),
                           ("eval_kpe_em2da", hot.eval_kpe_em2da),
                           ("eval_kpe_em2d", hot.eval_kpe_em2d)):
        # pycc captures the loop variable at decoration time, so make each
        # wrapper in a helper rather than exporting a Python closure.
        _export_kpe(cc, name, function)

    cc.compile()


def _export_kpe(cc: CC, name: str, function) -> None:
    """Register one k-perpendicular helper without changing its signature."""
    signature = "float64[:](float64[::1], float64, float64, float64[::1], float64[::1])"

    if name == "eval_kpe_std":
        @cc.export(name, signature)
        def wrapper(point, kpara, kperp, k, B):
            return hot.eval_kpe_std(point, kpara, kperp, k, B)
    elif name == "eval_kpe_em1d":
        @cc.export(name, signature)
        def wrapper(point, kpara, kperp, k, B):
            return hot.eval_kpe_em1d(point, kpara, kperp, k, B)
    elif name == "eval_kpe_em2da":
        @cc.export(name, signature)
        def wrapper(point, kpara, kperp, k, B):
            return hot.eval_kpe_em2da(point, kpara, kperp, k, B)
    else:
        @cc.export(name, signature)
        def wrapper(point, kpara, kperp, k, B):
            return hot.eval_kpe_em2d(point, kpara, kperp, k, B)


def build_all(output_dir: Path | None = None) -> None:
    """Build both extensions into *output_dir* (or this package directory)."""
    build_coldplasma(output_dir)
    build_lkplasma(output_dir)


def main() -> None:
    build_all()


if __name__ == "__main__":
    main()
