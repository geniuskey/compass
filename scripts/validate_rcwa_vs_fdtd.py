#!/usr/bin/env python3
"""RCWA vs FDTD cross-solver validation and cone illumination comparison.

Validates:
  1. grcwa vs fdtd_flaport at normal incidence (wavelength sweep)
  2. Cone illumination: direct vs F/2.0 vs F/2.0+CRA15°
  3. grcwa vs torcwa cone agreement

Usage:
    PYTHONPATH=. python3.11 scripts/validate_rcwa_vs_fdtd.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("validate_rcwa_vs_fdtd")

# ---------------------------------------------------------------------------
# Common config
# ---------------------------------------------------------------------------

PIXEL_CONFIG = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "layers": {
            "air": {"thickness": 1.0, "material": "air"},
            "microlens": {
                "enabled": True,
                "height": 0.6,
                "radius_x": 0.48,
                "radius_y": 0.48,
                "material": "polymer_n1p56",
                "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
                "shift": {"mode": "auto_cra", "cra_deg": 0.0},
                "gap": 0.0,
            },
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "pattern": "bayer_rggb",
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                "grid": {"enabled": True, "width": 0.05, "height": 0.6, "material": "tungsten"},
            },
            "barl": {
                "layers": [
                    {"thickness": 0.010, "material": "sio2"},
                    {"thickness": 0.025, "material": "hfo2"},
                    {"thickness": 0.015, "material": "sio2"},
                    {"thickness": 0.030, "material": "si3n4"},
                ],
            },
            "silicon": {
                "thickness": 3.0,
                "material": "silicon",
                "photodiode": {"position": [0.0, 0.0, 0.5], "size": [0.7, 0.7, 2.0]},
                "dti": {"enabled": True, "width": 0.1, "depth": 3.0, "material": "sio2"},
            },
        },
        "bayer_map": [["R", "G"], ["G", "B"]],
    },
    "compute": {"backend": "cpu"},
}


def make_sweep_source(theta_deg: float = 0.0, phi_deg: float = 0.0) -> dict:
    """Create wavelength sweep source config."""
    return {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.02}},
        "angle": {"theta_deg": theta_deg, "phi_deg": phi_deg},
        "polarization": "unpolarized",
    }


def make_cone_source(
    cra_deg: float = 0.0,
    f_number: float = 2.0,
    n_points: int = 19,
) -> dict:
    """Create cone illumination source config."""
    return {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.02}},
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "unpolarized",
        "cone": {
            "cra_deg": cra_deg,
            "f_number": f_number,
            "sampling": {"type": "fibonacci", "n_points": n_points},
            "weighting": "cosine",
        },
    }


# ---------------------------------------------------------------------------
# Experiment 1: RCWA (grcwa) vs FDTD (flaport) — normal incidence
# ---------------------------------------------------------------------------


def run_experiment_1() -> dict:
    """Compare grcwa vs fdtd_flaport at normal incidence."""
    from compass.runners.single_run import SingleRunner

    logger.info("=" * 70)
    logger.info("Experiment 1: grcwa vs fdtd_flaport — normal incidence sweep")
    logger.info("=" * 70)

    source = make_sweep_source(theta_deg=0.0)

    # grcwa
    grcwa_config = {
        **PIXEL_CONFIG,
        "solver": {
            "name": "grcwa",
            "type": "rcwa",
            "params": {"fourier_order": [5, 5], "dtype": "complex128"},
        },
        "source": source,
    }
    logger.info("Running grcwa [5,5]...")
    grcwa_result = SingleRunner.run(grcwa_config)

    # fdtd_flaport
    fdtd_config = {
        **PIXEL_CONFIG,
        "solver": {
            "name": "fdtd_flaport",
            "type": "fdtd",
            "params": {"grid_spacing": 0.015, "runtime": 500, "pml_layers": 20},
        },
        "source": source,
    }
    logger.info("Running fdtd_flaport (dx=0.015um, 500fs, pml=20)...")
    fdtd_result = SingleRunner.run(fdtd_config)

    # Compare
    wl = grcwa_result.wavelengths
    A_grcwa = grcwa_result.absorption
    A_fdtd = fdtd_result.absorption
    R_grcwa = grcwa_result.reflection
    R_fdtd = fdtd_result.reflection
    T_grcwa = grcwa_result.transmission
    T_fdtd = fdtd_result.transmission

    logger.info("")
    logger.info(
        f"{'WL(nm)':>8} | {'A_grcwa':>8} {'A_fdtd':>8} {'|dA|':>8} | "
        f"{'R_grcwa':>8} {'R_fdtd':>8} | {'T_grcwa':>8} {'T_fdtd':>8}"
    )
    logger.info("-" * 85)

    max_dA = 0.0
    for i, w in enumerate(wl):
        dA = abs(A_grcwa[i] - A_fdtd[i])
        max_dA = max(max_dA, dA)
        logger.info(
            f"{w * 1000:8.0f} | {A_grcwa[i]:8.4f} {A_fdtd[i]:8.4f} {dA:8.4f} | "
            f"{R_grcwa[i]:8.4f} {R_fdtd[i]:8.4f} | {T_grcwa[i]:8.4f} {T_fdtd[i]:8.4f}"
        )

    logger.info("")
    logger.info(f"Max |A_grcwa - A_fdtd| = {max_dA:.4f}")
    passed = max_dA < 0.05
    status = "PASS" if passed else "FAIL"
    logger.info(f"Experiment 1: {status} (threshold: 0.05)")

    return {
        "wavelengths": wl,
        "grcwa": {"R": R_grcwa, "T": T_grcwa, "A": A_grcwa},
        "fdtd": {"R": R_fdtd, "T": T_fdtd, "A": A_fdtd},
        "max_dA": max_dA,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Cone illumination comparison
# ---------------------------------------------------------------------------


def run_experiment_2() -> dict:
    """Compare direct vs cone illumination using grcwa."""
    from compass.runners.cone_runner import ConeIlluminationRunner
    from compass.runners.single_run import SingleRunner

    logger.info("")
    logger.info("=" * 70)
    logger.info("Experiment 2: Cone illumination — direct vs F/2.0 vs F/2.0+CRA15°")
    logger.info("=" * 70)

    # Direct (normal incidence)
    direct_config = {
        **PIXEL_CONFIG,
        "solver": {
            "name": "grcwa",
            "type": "rcwa",
            "params": {"fourier_order": [3, 3], "dtype": "complex128"},
        },
        "source": make_sweep_source(theta_deg=0.0),
    }
    logger.info("Running grcwa direct (θ=0°)...")
    direct_result = SingleRunner.run(direct_config)

    # Cone F/2.0 CRA=0°
    cone_0_config = {
        **PIXEL_CONFIG,
        "solver": {
            "name": "grcwa",
            "type": "rcwa",
            "params": {"fourier_order": [3, 3], "dtype": "complex128"},
        },
        "source": make_cone_source(cra_deg=0.0, f_number=2.0, n_points=19),
    }
    logger.info("Running grcwa cone F/2.0 CRA=0° (19 points)...")
    cone_0_result = ConeIlluminationRunner.run(cone_0_config)

    # Cone F/2.0 CRA=15°
    cone_15_config = {
        **PIXEL_CONFIG,
        "solver": {
            "name": "grcwa",
            "type": "rcwa",
            "params": {"fourier_order": [3, 3], "dtype": "complex128"},
        },
        "source": make_cone_source(cra_deg=15.0, f_number=2.0, n_points=19),
    }
    logger.info("Running grcwa cone F/2.0 CRA=15° (19 points)...")
    cone_15_result = ConeIlluminationRunner.run(cone_15_config)

    wl = direct_result.wavelengths
    logger.info("")
    logger.info(f"{'WL(nm)':>8} | {'A_direct':>9} {'A_cone0':>9} {'A_cone15':>9}")
    logger.info("-" * 48)

    for i, w in enumerate(wl):
        logger.info(
            f"{w * 1000:8.0f} | {direct_result.absorption[i]:9.4f} "
            f"{cone_0_result.absorption[i]:9.4f} "
            f"{cone_15_result.absorption[i]:9.4f}"
        )

    return {
        "wavelengths": wl,
        "direct": {
            "R": direct_result.reflection,
            "T": direct_result.transmission,
            "A": direct_result.absorption,
        },
        "cone_0": {
            "R": cone_0_result.reflection,
            "T": cone_0_result.transmission,
            "A": cone_0_result.absorption,
        },
        "cone_15": {
            "R": cone_15_result.reflection,
            "T": cone_15_result.transmission,
            "A": cone_15_result.absorption,
        },
    }


# ---------------------------------------------------------------------------
# Experiment 3: grcwa vs torcwa cone agreement
# ---------------------------------------------------------------------------


def run_experiment_3() -> dict:
    """Compare grcwa vs torcwa with cone illumination."""
    from compass.runners.cone_runner import ConeIlluminationRunner

    logger.info("")
    logger.info("=" * 70)
    logger.info("Experiment 3: grcwa vs torcwa cone F/2.0 CRA=0°")
    logger.info("=" * 70)

    cone_source = make_cone_source(cra_deg=0.0, f_number=2.0, n_points=19)

    # grcwa cone
    grcwa_cone_config = {
        **PIXEL_CONFIG,
        "solver": {
            "name": "grcwa",
            "type": "rcwa",
            "params": {"fourier_order": [3, 3], "dtype": "complex128"},
        },
        "source": cone_source,
    }
    logger.info("Running grcwa cone F/2.0...")
    grcwa_result = ConeIlluminationRunner.run(grcwa_cone_config)

    # torcwa cone
    torcwa_cone_config = {
        **PIXEL_CONFIG,
        "solver": {
            "name": "torcwa",
            "type": "rcwa",
            "params": {"fourier_order": [3, 3], "dtype": "complex64"},
        },
        "source": cone_source,
    }
    logger.info("Running torcwa cone F/2.0...")
    torcwa_result = ConeIlluminationRunner.run(torcwa_cone_config)

    wl = grcwa_result.wavelengths
    A_grcwa = grcwa_result.absorption
    A_torcwa = torcwa_result.absorption

    max_dA = float(np.max(np.abs(A_grcwa - A_torcwa)))

    logger.info("")
    logger.info(f"{'WL(nm)':>8} | {'A_grcwa':>8} {'A_torcwa':>9} {'|dA|':>8}")
    logger.info("-" * 42)

    for i, w in enumerate(wl):
        dA = abs(A_grcwa[i] - A_torcwa[i])
        logger.info(f"{w * 1000:8.0f} | {A_grcwa[i]:8.4f} {A_torcwa[i]:9.4f} {dA:8.4f}")

    logger.info("")
    logger.info(f"Max |A_grcwa - A_torcwa| (cone) = {max_dA:.4f}")
    passed = max_dA < 0.05
    status = "PASS" if passed else "FAIL"
    logger.info(f"Experiment 3: {status} (threshold: 0.05)")

    return {
        "wavelengths": wl,
        "grcwa": {"R": grcwa_result.reflection, "T": grcwa_result.transmission, "A": A_grcwa},
        "torcwa": {"R": torcwa_result.reflection, "T": torcwa_result.transmission, "A": A_torcwa},
        "max_dA": max_dA,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("COMPASS RCWA vs FDTD Cross-Solver Validation")
    logger.info("=" * 70)

    results = {}

    # Experiment 1: RCWA vs FDTD
    try:
        results["exp1"] = run_experiment_1()
    except Exception as e:
        logger.error(f"Experiment 1 failed: {e}")
        results["exp1"] = {"passed": False, "error": str(e)}

    # Experiment 2: Cone illumination
    try:
        results["exp2"] = run_experiment_2()
    except Exception as e:
        logger.error(f"Experiment 2 failed: {e}")
        results["exp2"] = {"error": str(e)}

    # Experiment 3: grcwa vs torcwa cone
    try:
        results["exp3"] = run_experiment_3()
    except Exception as e:
        logger.error(f"Experiment 3 failed: {e}")
        results["exp3"] = {"passed": False, "error": str(e)}

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)

    all_passed = True
    for name, res in results.items():
        if "error" in res:
            logger.info(f"  {name}: ERROR — {res['error']}")
            all_passed = False
        elif "passed" in res:
            status = "PASS" if res["passed"] else "FAIL"
            logger.info(f"  {name}: {status}")
            if not res["passed"]:
                all_passed = False
        else:
            logger.info(f"  {name}: completed (info only)")

    if all_passed:
        logger.info("\nAll validation experiments PASSED.")
    else:
        logger.info("\nSome experiments FAILED — see details above.")

    # Save results to JSON
    output_dir = Path(__file__).parent.parent / "results" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rcwa_vs_fdtd.json"

    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    serializable = {}
    for key, val in results.items():
        if isinstance(val, dict):
            serializable[key] = {
                k: (
                    {kk: _to_serializable(vv) for kk, vv in v.items()}
                    if isinstance(v, dict)
                    else _to_serializable(v)
                )
                for k, v in val.items()
            }
        else:
            serializable[key] = _to_serializable(val)

    output_path.write_text(json.dumps(serializable, indent=2))
    logger.info(f"\nResults saved to {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
