#!/usr/bin/env python3
"""Convergence study script for COMPASS RCWA solvers.

Runs parametric sweeps of Fourier order, lens slicing, and grid resolution
to determine converged simulation parameters. Outputs formatted tables to
stdout and JSON results to disk.

Usage:
    PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep fourier_order_grcwa
    PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep fourier_order_torcwa
    PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep n_lens_slices
    PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep grid_resolution
    PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep full_spectrum
    PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep fourier_per_color_torcwa
    PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep fourier_per_color_grcwa
    PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep per_color_spectrum
    PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep cross_solver_per_color
    PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep angle_per_color
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from compass.geometry.pixel_stack import PixelStack
from compass.materials.database import MaterialDB
from compass.solvers.base import SolverFactory

logger = logging.getLogger(__name__)

# Default pixel configuration (matches default_bsi_1um.yaml)
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
                "grid": {
                    "enabled": True,
                    "width": 0.05,
                    "height": 0.6,
                    "material": "tungsten",
                },
            },
            "barl": {
                "layers": [
                    {"thickness": 0.010, "material": "sio2"},
                    {"thickness": 0.025, "material": "hfo2"},
                    {"thickness": 0.015, "material": "sio2"},
                    {"thickness": 0.030, "material": "si3n4"},
                ]
            },
            "silicon": {
                "thickness": 3.0,
                "material": "silicon",
                "photodiode": {"position": [0.0, 0.0, 0.5], "size": [0.7, 0.7, 2.0]},
                "dti": {"enabled": True, "width": 0.1, "depth": 3.0, "material": "sio2"},
            },
        },
        "bayer_map": [["R", "G"], ["G", "B"]],
    }
}

CONVERGENCE_THRESHOLD = 0.001


def build_source_config(wavelength: float) -> dict:
    """Build source config for a single wavelength.

    Args:
        wavelength: Wavelength in um.

    Returns:
        Source configuration dictionary.
    """
    return {
        "type": "planewave",
        "wavelength": {"mode": "single", "value": wavelength},
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "unpolarized",
    }


def build_source_config_spectrum(start: float, end: float, num: int) -> dict:
    """Build source config for a wavelength sweep.

    Args:
        start: Start wavelength in um.
        end: End wavelength in um.
        num: Number of wavelength points.

    Returns:
        Source configuration dictionary.
    """
    wavelengths = np.linspace(start, end, num).tolist()
    return {
        "type": "planewave",
        "wavelength": {"mode": "list", "values": wavelengths},
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "unpolarized",
    }


def build_source_config_multi_wavelength(
    wavelengths: list[float], theta_deg: float = 0.0, phi_deg: float = 0.0,
) -> dict:
    """Build source config for multiple specific wavelengths.

    Args:
        wavelengths: List of wavelengths in um.
        theta_deg: Polar incidence angle in degrees.
        phi_deg: Azimuthal incidence angle in degrees.

    Returns:
        Source configuration dictionary.
    """
    return {
        "type": "planewave",
        "wavelength": {"mode": "list", "values": wavelengths},
        "angle": {"theta_deg": theta_deg, "phi_deg": phi_deg},
        "polarization": "unpolarized",
    }


def run_single_sim(
    solver_name: str,
    fourier_order: list[int],
    n_lens_slices: int,
    grid_multiplier: int,
    source_config: dict,
    pixel_config: dict | None = None,
) -> tuple[float, float, float, float]:
    """Run a single simulation and return R, T, A, elapsed time.

    Args:
        solver_name: Solver name ("grcwa" or "torcwa").
        fourier_order: Fourier order as [N, N] or [nG, nG].
        n_lens_slices: Number of microlens staircase slices.
        grid_multiplier: Grid resolution multiplier.
        source_config: Source configuration dictionary.
        pixel_config: Pixel configuration (defaults to PIXEL_CONFIG).

    Returns:
        Tuple of (R, T, A, elapsed_seconds).
    """
    material_db = MaterialDB()
    pixel_stack = PixelStack(pixel_config or PIXEL_CONFIG, material_db)

    solver_config = {
        "name": solver_name,
        "type": "rcwa",
        "params": {
            "fourier_order": fourier_order,
            "dtype": "complex128",
            "n_lens_slices": n_lens_slices,
            "grid_multiplier": grid_multiplier,
        },
    }

    solver = SolverFactory.create(solver_name, solver_config, "cpu")
    solver.setup_geometry(pixel_stack)
    solver.setup_source(source_config)

    t0 = time.perf_counter()
    result = solver.run()
    elapsed = time.perf_counter() - t0

    R = float(result.reflection[0])
    T = float(result.transmission[0])
    A = float(result.absorption[0])

    return R, T, A, elapsed


def run_spectrum_sim(
    solver_name: str,
    fourier_order: list[int],
    n_lens_slices: int,
    grid_multiplier: int,
    source_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run a full spectrum simulation.

    Args:
        solver_name: Solver name.
        fourier_order: Fourier order.
        n_lens_slices: Number of microlens staircase slices.
        grid_multiplier: Grid resolution multiplier.
        source_config: Source configuration dictionary.

    Returns:
        Tuple of (wavelengths, R_array, T_array, A_array, elapsed_seconds).
    """
    material_db = MaterialDB()
    pixel_stack = PixelStack(PIXEL_CONFIG, material_db)

    solver_config = {
        "name": solver_name,
        "type": "rcwa",
        "params": {
            "fourier_order": fourier_order,
            "dtype": "complex128",
            "n_lens_slices": n_lens_slices,
            "grid_multiplier": grid_multiplier,
        },
    }

    solver = SolverFactory.create(solver_name, solver_config, "cpu")
    solver.setup_geometry(pixel_stack)
    solver.setup_source(source_config)

    t0 = time.perf_counter()
    result = solver.run()
    elapsed = time.perf_counter() - t0

    return result.wavelengths, result.reflection, result.transmission, result.absorption, elapsed


def run_sim_with_qe(
    solver_name: str,
    fourier_order: list[int],
    n_lens_slices: int,
    grid_multiplier: int,
    source_config: dict,
    pixel_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], float]:
    """Run simulation returning full results including per-pixel QE.

    Args:
        solver_name: Solver name ("grcwa" or "torcwa").
        fourier_order: Fourier order as [N, N] or [nG, nG].
        n_lens_slices: Number of microlens staircase slices.
        grid_multiplier: Grid resolution multiplier.
        source_config: Source configuration dictionary.
        pixel_config: Pixel configuration (defaults to PIXEL_CONFIG).

    Returns:
        Tuple of (wavelengths, R, T, A, qe_per_pixel, elapsed_seconds).
    """
    material_db = MaterialDB()
    pixel_stack = PixelStack(pixel_config or PIXEL_CONFIG, material_db)

    solver_config = {
        "name": solver_name,
        "type": "rcwa",
        "params": {
            "fourier_order": fourier_order,
            "dtype": "complex128",
            "n_lens_slices": n_lens_slices,
            "grid_multiplier": grid_multiplier,
        },
    }

    solver = SolverFactory.create(solver_name, solver_config, "cpu")
    solver.setup_geometry(pixel_stack)
    solver.setup_source(source_config)

    t0 = time.perf_counter()
    result = solver.run()
    elapsed = time.perf_counter() - t0

    return (
        result.wavelengths,
        result.reflection,
        result.transmission,
        result.absorption,
        result.qe_per_pixel,
        elapsed,
    )


def extract_color_qe(qe_per_pixel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Group per-pixel QE by color channel (average same-color pixels).

    For a 2x2 RGGB Bayer, the two G pixels are averaged.

    Args:
        qe_per_pixel: Dict mapping pixel names (e.g. "R_0_0") to QE arrays.

    Returns:
        Dict mapping color names ("B", "G", "R") to averaged QE arrays.
    """
    color_qe: dict[str, np.ndarray] = {}
    color_count: dict[str, int] = {}
    for name, qe in qe_per_pixel.items():
        color = name.split("_")[0]
        if color not in color_qe:
            color_qe[color] = np.zeros_like(qe)
            color_count[color] = 0
        color_qe[color] += qe
        color_count[color] += 1
    return {c: color_qe[c] / color_count[c] for c in sorted(color_qe)}


def _pixel_config_with_cra(cra_deg: float) -> dict:
    """Create pixel config with microlens CRA shift set.

    Args:
        cra_deg: Chief Ray Angle in degrees for microlens shift.

    Returns:
        Deep-copied pixel config with updated CRA.
    """
    config = json.loads(json.dumps(PIXEL_CONFIG))
    config["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = cra_deg
    return config


def check_convergence(results: list[dict]) -> dict | None:
    """Check if convergence criterion is met: |delta_A| < threshold for 2 consecutive steps.

    Args:
        results: List of result dicts with "delta_A" keys.

    Returns:
        Dict with convergence info if converged, else None.
    """
    consecutive = 0
    for i, r in enumerate(results):
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
            if consecutive >= 2:
                return {"index": i - 1, "result": results[i - 1]}
        else:
            consecutive = 0
    return None


def print_fourier_table(results: list[dict], solver_name: str) -> None:
    """Print formatted convergence table for Fourier order sweep.

    Args:
        results: List of result dictionaries.
        solver_name: Solver name for the header.
    """
    print(f"\n=== Fourier Order Convergence ({solver_name}) ===")
    print(
        f"{'nG':>6} | {'Harmonics':>9} | {'R':>9} | {'T':>9} | {'A':>9} "
        f"| {'dA':>9} | {'Time (s)':>9} | Converged"
    )
    print(
        "-" * 6
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
    )

    consecutive = 0
    for r in results:
        delta_str = f"{r['delta_A']:>9.5f}" if r["delta_A"] is not None else "        —"
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 2:
            conv_mark = " **"
        elif consecutive == 1:
            conv_mark = "  *"
        else:
            conv_mark = "   "
        print(
            f"{r['nG']:>6} | {r['harmonics']:>9} | {r['R']:>9.5f} | {r['T']:>9.5f} "
            f"| {r['A']:>9.5f} | {delta_str} | {r['time_s']:>9.2f} | {conv_mark}"
        )


def print_parameter_table(results: list[dict], param_name: str, param_key: str, title: str) -> None:
    """Print formatted convergence table for a parameter sweep.

    Args:
        results: List of result dictionaries.
        param_name: Display name for the parameter column.
        param_key: Key in the result dict for the parameter value.
        title: Table title.
    """
    print(f"\n=== {title} ===")
    print(
        f"{param_name:>10} | {'R':>9} | {'T':>9} | {'A':>9} "
        f"| {'dA':>9} | {'Time (s)':>9} | Converged"
    )
    print(
        "-" * 10
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
        + "-+-"
        + "-" * 9
    )

    consecutive = 0
    for r in results:
        delta_str = f"{r['delta_A']:>9.5f}" if r["delta_A"] is not None else "        —"
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= 2:
            conv_mark = " **"
        elif consecutive == 1:
            conv_mark = "  *"
        else:
            conv_mark = "   "
        print(
            f"{r[param_key]:>10} | {r['R']:>9.5f} | {r['T']:>9.5f} "
            f"| {r['A']:>9.5f} | {delta_str} | {r['time_s']:>9.2f} | {conv_mark}"
        )


def print_spectrum_table(
    wavelengths: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    A: np.ndarray,
) -> None:
    """Print formatted spectrum results table.

    Args:
        wavelengths: Wavelength array in um.
        R: Reflection array.
        T: Transmission array.
        A: Absorption array.
    """
    print("\n=== Full Spectrum Results (grcwa) ===")
    print(f"{'lambda (um)':>12} | {'R':>9} | {'T':>9} | {'A':>9} | {'R+T+A':>9}")
    print("-" * 12 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+-" + "-" * 9)

    for i in range(len(wavelengths)):
        total = R[i] + T[i] + A[i]
        print(
            f"{wavelengths[i]:>12.4f} | {R[i]:>9.5f} | {T[i]:>9.5f} | {A[i]:>9.5f} | {total:>9.5f}"
        )


def save_json(data: dict, output_dir: str, filename: str) -> None:
    """Save results dict to JSON file.

    Args:
        data: Results dictionary.
        output_dir: Output directory path.
        filename: Output filename.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to {filepath}")


def sweep_fourier_order_grcwa(output_dir: str) -> None:
    """Run Fourier order convergence sweep for grcwa solver.

    Sweeps nG values and tracks R, T, A convergence.

    Args:
        output_dir: Directory for JSON output.
    """
    nG_values = [9, 25, 49, 81, 121, 169, 225, 289, 361, 441, 529, 625]
    wavelength = 0.55
    source_config = build_source_config(wavelength)

    results = []
    prev_A = None

    for nG in nG_values:
        print(f"  Running grcwa nG={nG}...", end=" ", flush=True)
        R, T, A, elapsed = run_single_sim(
            "grcwa",
            [nG, nG],
            n_lens_slices=30,
            grid_multiplier=3,
            source_config=source_config,
        )
        delta_A = (A - prev_A) if prev_A is not None else None
        results.append(
            {
                "nG": nG,
                "harmonics": nG,
                "R": R,
                "T": T,
                "A": A,
                "delta_A": delta_A,
                "time_s": elapsed,
            }
        )
        prev_A = A
        print(f"A={A:.5f}, time={elapsed:.2f}s")

    # Mark convergence
    consecutive = 0
    for r in results:
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
        else:
            consecutive = 0
        r["converged"] = consecutive >= 2

    print_fourier_table(results, "grcwa")

    # Find convergence point
    converged_at = None
    consecutive = 0
    for i, r in enumerate(results):
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
            if consecutive >= 2:
                converged_at = {"nG": results[i - 1]["nG"], "index": i - 1}
                break
        else:
            consecutive = 0

    json_data = {
        "sweep_type": "fourier_order_grcwa",
        "wavelength": wavelength,
        "timestamp": datetime.now(UTC).isoformat(),
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "converged_at": converged_at,
        "results": results,
    }
    save_json(json_data, output_dir, "fourier_order_grcwa.json")


def sweep_fourier_order_torcwa(output_dir: str) -> None:
    """Run Fourier order convergence sweep for torcwa solver.

    Sweeps [N,N] Fourier order values and tracks R, T, A convergence.

    Args:
        output_dir: Directory for JSON output.
    """
    N_values = [3, 5, 7, 9, 11, 13, 15, 17, 21, 25]
    wavelength = 0.55
    source_config = build_source_config(wavelength)

    results = []
    prev_A = None

    for N in N_values:
        harmonics = (2 * N + 1) ** 2
        print(f"  Running torcwa N={N} (harmonics={harmonics})...", end=" ", flush=True)
        R, T, A, elapsed = run_single_sim(
            "torcwa",
            [N, N],
            n_lens_slices=30,
            grid_multiplier=3,
            source_config=source_config,
        )
        delta_A = (A - prev_A) if prev_A is not None else None
        results.append(
            {
                "nG": N,
                "harmonics": harmonics,
                "R": R,
                "T": T,
                "A": A,
                "delta_A": delta_A,
                "time_s": elapsed,
            }
        )
        prev_A = A
        print(f"A={A:.5f}, time={elapsed:.2f}s")

    # Mark convergence
    consecutive = 0
    for r in results:
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
        else:
            consecutive = 0
        r["converged"] = consecutive >= 2

    print_fourier_table(results, "torcwa")

    converged_at = None
    consecutive = 0
    for i, r in enumerate(results):
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
            if consecutive >= 2:
                converged_at = {"nG": results[i - 1]["nG"], "index": i - 1}
                break
        else:
            consecutive = 0

    json_data = {
        "sweep_type": "fourier_order_torcwa",
        "wavelength": wavelength,
        "timestamp": datetime.now(UTC).isoformat(),
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "converged_at": converged_at,
        "results": results,
    }
    save_json(json_data, output_dir, "fourier_order_torcwa.json")


def sweep_n_lens_slices(output_dir: str, fourier_order: int) -> None:
    """Run microlens slice count convergence sweep.

    Uses grcwa solver with specified Fourier order.

    Args:
        output_dir: Directory for JSON output.
        fourier_order: grcwa nG value to use.
    """
    slice_values = [5, 10, 15, 20, 30, 50, 80]
    wavelength = 0.55
    source_config = build_source_config(wavelength)

    results = []
    prev_A = None

    for n_slices in slice_values:
        print(f"  Running grcwa n_lens_slices={n_slices}...", end=" ", flush=True)
        R, T, A, elapsed = run_single_sim(
            "grcwa",
            [fourier_order, fourier_order],
            n_lens_slices=n_slices,
            grid_multiplier=3,
            source_config=source_config,
        )
        delta_A = (A - prev_A) if prev_A is not None else None
        results.append(
            {
                "n_lens_slices": n_slices,
                "R": R,
                "T": T,
                "A": A,
                "delta_A": delta_A,
                "time_s": elapsed,
            }
        )
        prev_A = A
        print(f"A={A:.5f}, time={elapsed:.2f}s")

    # Mark convergence
    consecutive = 0
    for r in results:
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
        else:
            consecutive = 0
        r["converged"] = consecutive >= 2

    print_parameter_table(
        results, "n_slices", "n_lens_slices", "Microlens Slice Convergence (grcwa)"
    )

    converged_at = None
    consecutive = 0
    for i, r in enumerate(results):
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
            if consecutive >= 2:
                converged_at = {
                    "n_lens_slices": results[i - 1]["n_lens_slices"],
                    "index": i - 1,
                }
                break
        else:
            consecutive = 0

    json_data = {
        "sweep_type": "n_lens_slices",
        "wavelength": wavelength,
        "fourier_order": fourier_order,
        "timestamp": datetime.now(UTC).isoformat(),
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "converged_at": converged_at,
        "results": results,
    }
    save_json(json_data, output_dir, "n_lens_slices.json")


def sweep_grid_resolution(output_dir: str, fourier_order: int, n_lens_slices: int) -> None:
    """Run grid resolution convergence sweep.

    Uses grcwa solver with specified Fourier order and lens slices.

    Args:
        output_dir: Directory for JSON output.
        fourier_order: grcwa nG value to use.
        n_lens_slices: Number of microlens staircase slices.
    """
    multiplier_values = [2, 3, 4, 5, 6]
    wavelength = 0.55
    source_config = build_source_config(wavelength)

    results = []
    prev_A = None

    for mult in multiplier_values:
        print(f"  Running grcwa grid_multiplier={mult}...", end=" ", flush=True)
        R, T, A, elapsed = run_single_sim(
            "grcwa",
            [fourier_order, fourier_order],
            n_lens_slices=n_lens_slices,
            grid_multiplier=mult,
            source_config=source_config,
        )
        delta_A = (A - prev_A) if prev_A is not None else None
        results.append(
            {
                "grid_multiplier": mult,
                "R": R,
                "T": T,
                "A": A,
                "delta_A": delta_A,
                "time_s": elapsed,
            }
        )
        prev_A = A
        print(f"A={A:.5f}, time={elapsed:.2f}s")

    # Mark convergence
    consecutive = 0
    for r in results:
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
        else:
            consecutive = 0
        r["converged"] = consecutive >= 2

    print_parameter_table(
        results, "grid_mult", "grid_multiplier", "Grid Resolution Convergence (grcwa)"
    )

    converged_at = None
    consecutive = 0
    for i, r in enumerate(results):
        if r["delta_A"] is not None and abs(r["delta_A"]) < CONVERGENCE_THRESHOLD:
            consecutive += 1
            if consecutive >= 2:
                converged_at = {
                    "grid_multiplier": results[i - 1]["grid_multiplier"],
                    "index": i - 1,
                }
                break
        else:
            consecutive = 0

    json_data = {
        "sweep_type": "grid_resolution",
        "wavelength": wavelength,
        "fourier_order": fourier_order,
        "n_lens_slices": n_lens_slices,
        "timestamp": datetime.now(UTC).isoformat(),
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "converged_at": converged_at,
        "results": results,
    }
    save_json(json_data, output_dir, "grid_resolution.json")


def sweep_full_spectrum(
    output_dir: str, fourier_order: int, n_lens_slices: int, grid_multiplier: int
) -> None:
    """Run full visible spectrum simulation at converged parameters.

    Args:
        output_dir: Directory for JSON output.
        fourier_order: grcwa nG value to use.
        n_lens_slices: Number of microlens staircase slices.
        grid_multiplier: Grid resolution multiplier.
    """
    source_config = build_source_config_spectrum(0.4, 0.7, 31)

    print(
        f"  Running grcwa full spectrum (400-700nm, 31 pts), "
        f"nG={fourier_order}, slices={n_lens_slices}, grid_mult={grid_multiplier}..."
    )

    wavelengths, R, T, A, elapsed = run_spectrum_sim(
        "grcwa",
        [fourier_order, fourier_order],
        n_lens_slices=n_lens_slices,
        grid_multiplier=grid_multiplier,
        source_config=source_config,
    )

    print(f"  Completed in {elapsed:.2f}s")

    print_spectrum_table(wavelengths, R, T, A)

    spectrum_results = []
    for i in range(len(wavelengths)):
        spectrum_results.append(
            {
                "wavelength": float(wavelengths[i]),
                "R": float(R[i]),
                "T": float(T[i]),
                "A": float(A[i]),
            }
        )

    json_data = {
        "sweep_type": "full_spectrum",
        "fourier_order": fourier_order,
        "n_lens_slices": n_lens_slices,
        "grid_multiplier": grid_multiplier,
        "timestamp": datetime.now(UTC).isoformat(),
        "total_time_s": elapsed,
        "results": spectrum_results,
    }
    save_json(json_data, output_dir, "full_spectrum.json")


# -- Per-color QE convergence sweeps --
#
# Methodology: "uniform color filter" approach.
# The per-pixel QE from a Bayer simulation distributes total absorption
# equally because silicon eps_imag is uniform across all PD regions.
# To get true per-color QE, we run separate simulations with a uniform
# color filter (all pixels use the same CF material) at each color's
# peak wavelength. The total absorption A then equals the QE through
# that specific color filter — the standard CIS measurement.

# Color filter materials and their passband peak wavelengths (um)
CF_MATERIALS = {"R": "cf_red", "G": "cf_green", "B": "cf_blue"}
COLOR_PEAK_WL = {"B": 0.45, "G": 0.53, "R": 0.62}


def _pixel_config_uniform_cf(cf_color: str) -> dict:
    """Create pixel config with uniform color filter of a single color.

    All pixels in the Bayer pattern use the same CF material. This isolates
    the spectral response of a single color channel for accurate QE measurement.

    Args:
        cf_color: Color channel ("R", "G", or "B").

    Returns:
        Deep-copied pixel config with uniform CF.
    """
    config = json.loads(json.dumps(PIXEL_CONFIG))
    cf_material = CF_MATERIALS[cf_color]
    config["pixel"]["layers"]["color_filter"]["materials"] = {
        "R": cf_material, "G": cf_material, "B": cf_material,
    }
    return config


def print_per_color_fourier_table(results: list[dict], solver_name: str) -> None:
    """Print per-color QE convergence table for Fourier order sweep.

    Args:
        results: List of result dictionaries with per-color QE.
        solver_name: Solver name for the header.
    """
    print(f"\n=== Per-Color QE Convergence ({solver_name}, uniform CF) ===")
    print(
        f"{'nG':>6} | {'Harm':>6} | {'QE_R@620':>9} | {'QE_G@530':>9} "
        f"| {'QE_B@450':>9} | {'dQE_R':>9} | {'dQE_G':>9} | {'dQE_B':>9} "
        f"| {'Time':>8}"
    )
    print("-" * 95)

    for r in results:
        dR = f"{r['delta_QE_R']:>9.5f}" if r["delta_QE_R"] is not None else "        —"
        dG = f"{r['delta_QE_G']:>9.5f}" if r["delta_QE_G"] is not None else "        —"
        dB = f"{r['delta_QE_B']:>9.5f}" if r["delta_QE_B"] is not None else "        —"
        print(
            f"{r['nG']:>6} | {r['harmonics']:>6} | {r['QE_R']:>9.5f} | {r['QE_G']:>9.5f} "
            f"| {r['QE_B']:>9.5f} | {dR} | {dG} | {dB} "
            f"| {r['time_s']:>7.1f}s"
        )


def _run_per_color_fourier_sweep(
    solver_name: str,
    order_values: list[int],
    output_dir: str,
) -> None:
    """Run per-color QE Fourier order convergence sweep using uniform CF.

    For each Fourier order and each color (B, G, R), runs a simulation with
    a uniform color filter at the CF's peak wavelength. Total absorption A
    equals the QE through that color filter.

    Args:
        solver_name: "grcwa" or "torcwa".
        order_values: Fourier order values to sweep.
        output_dir: Directory for JSON output.
    """
    is_torcwa = solver_name == "torcwa"

    results = []
    prev_qe: dict[str, float | None] = {"R": None, "G": None, "B": None}

    for order in order_values:
        harmonics = (2 * order + 1) ** 2 if is_torcwa else order
        fo = [order, order]
        label = f"N={order}" if is_torcwa else f"nG={order}"
        print(f"  {solver_name} {label} ({harmonics} harmonics):")

        entry: dict = {"nG": order, "harmonics": harmonics}
        total_time = 0.0
        failed = False

        for color in ["B", "G", "R"]:
            wl = COLOR_PEAK_WL[color]
            pixel_config = _pixel_config_uniform_cf(color)
            source_config = build_source_config(wl)

            print(f"    {color} (λ={wl * 1000:.0f}nm)...", end=" ", flush=True)

            try:
                R, T, A, elapsed = run_single_sim(
                    solver_name, fo, n_lens_slices=30, grid_multiplier=3,
                    source_config=source_config,
                    pixel_config=pixel_config,
                )
            except Exception as e:
                print(f"FAILED: {e}")
                failed = True
                break

            a_val = float(A)
            delta = (a_val - prev_qe[color]) if prev_qe[color] is not None else None
            entry[f"QE_{color}"] = a_val
            entry[f"delta_QE_{color}"] = delta
            entry[f"R_{color}"] = float(R)
            entry[f"T_{color}"] = float(T)
            total_time += elapsed
            prev_qe[color] = a_val

            print(f"A={a_val:.5f} R={R:.5f} T={T:.5f} ({elapsed:.1f}s)")

        if failed:
            continue

        entry["time_s"] = total_time
        results.append(entry)

    print_per_color_fourier_table(results, solver_name)

    filename = f"fourier_per_color_{solver_name}.json"
    json_data = {
        "sweep_type": f"fourier_per_color_{solver_name}",
        "methodology": "uniform_cf",
        "color_wavelengths": COLOR_PEAK_WL,
        "timestamp": datetime.now(UTC).isoformat(),
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "results": results,
    }
    save_json(json_data, output_dir, filename)


def sweep_fourier_per_color_torcwa(output_dir: str) -> None:
    """Per-color QE Fourier order convergence sweep for torcwa (uniform CF)."""
    _run_per_color_fourier_sweep("torcwa", [3, 5, 7, 9], output_dir)


def sweep_fourier_per_color_grcwa(output_dir: str) -> None:
    """Per-color QE Fourier order convergence sweep for grcwa (uniform CF)."""
    _run_per_color_fourier_sweep("grcwa", [9, 25, 49, 81], output_dir)


def sweep_per_color_spectrum(output_dir: str) -> None:
    """Full visible spectrum per-color QE for both solvers (uniform CF).

    For each color, runs a full 400-700nm sweep with a uniform color filter.
    This produces classic R/G/B spectral response curves.

    Args:
        output_dir: Directory for JSON output.
    """
    all_results = {}

    for solver_name, fo in [("grcwa", [49, 49]), ("torcwa", [5, 5])]:
        print(f"\n  {solver_name} per-color spectrum (400-700nm, 31 pts):")
        solver_data: dict = {"fourier_order": fo, "colors": {}}
        total_time = 0.0

        for color in ["B", "G", "R"]:
            pixel_config = _pixel_config_uniform_cf(color)
            source_config = build_source_config_spectrum(0.4, 0.7, 31)

            print(f"    {color} filter...", end=" ", flush=True)

            wavelengths, R, T, A, _qe, elapsed = run_sim_with_qe(
                solver_name, fo, n_lens_slices=30, grid_multiplier=3,
                source_config=source_config,
                pixel_config=pixel_config,
            )

            color_results = []
            for i in range(len(wavelengths)):
                color_results.append({
                    "wavelength": float(wavelengths[i]),
                    "QE": float(A[i]),
                    "R": float(R[i]),
                    "T": float(T[i]),
                })

            peak_idx = int(np.argmax(A))
            peak_wl = float(wavelengths[peak_idx])
            peak_qe = float(A[peak_idx])
            mean_qe = float(np.mean(A))

            solver_data["colors"][color] = {
                "time_s": elapsed,
                "peak_wavelength_nm": peak_wl * 1000,
                "peak_qe": peak_qe,
                "mean_qe": mean_qe,
                "results": color_results,
            }
            total_time += elapsed

            print(
                f"peak QE={peak_qe:.4f} at {peak_wl * 1000:.0f}nm, "
                f"mean={mean_qe:.4f} ({elapsed:.1f}s)"
            )

        solver_data["total_time_s"] = total_time
        all_results[solver_name] = solver_data

    json_data = {
        "sweep_type": "per_color_spectrum",
        "methodology": "uniform_cf",
        "timestamp": datetime.now(UTC).isoformat(),
        "solvers": all_results,
    }
    save_json(json_data, output_dir, "per_color_spectrum.json")


def sweep_cross_solver_per_color(output_dir: str) -> None:
    """Cross-solver per-color QE comparison using uniform CF.

    For each color, runs both solvers at the CF peak wavelength with a
    uniform color filter to get true per-color QE.

    Args:
        output_dir: Directory for JSON output.
    """
    solver_results = {}

    for solver_name, fo in [("grcwa", [49, 49]), ("torcwa", [5, 5])]:
        print(f"\n  {solver_name}:")
        result: dict = {"fourier_order": fo}
        total_time = 0.0

        for color in ["B", "G", "R"]:
            wl = COLOR_PEAK_WL[color]
            pixel_config = _pixel_config_uniform_cf(color)
            source_config = build_source_config(wl)

            print(f"    {color} (λ={wl * 1000:.0f}nm)...", end=" ", flush=True)

            R, T, A, elapsed = run_single_sim(
                solver_name, fo, n_lens_slices=30, grid_multiplier=3,
                source_config=source_config,
                pixel_config=pixel_config,
            )

            result[f"QE_{color}"] = float(A)
            result[f"R_{color}"] = float(R)
            result[f"T_{color}"] = float(T)
            total_time += elapsed

            print(f"QE={A:.5f} R={R:.5f} ({elapsed:.1f}s)")

        result["time_s"] = total_time
        solver_results[solver_name] = result

    # Print comparison table
    print("\n=== Cross-Solver Per-Color QE (uniform CF) ===")
    print(f"{'':>8} | {'QE_R@620':>9} | {'QE_G@530':>9} | {'QE_B@450':>9}")
    print("-" * 50)
    for sn in ["grcwa", "torcwa"]:
        r = solver_results[sn]
        print(f"{sn:>8} | {r['QE_R']:>9.5f} | {r['QE_G']:>9.5f} | {r['QE_B']:>9.5f}")

    g = solver_results["grcwa"]
    t = solver_results["torcwa"]
    print(
        f"{'diff':>8} | {g['QE_R'] - t['QE_R']:>+9.5f} | "
        f"{g['QE_G'] - t['QE_G']:>+9.5f} | {g['QE_B'] - t['QE_B']:>+9.5f}"
    )

    json_data = {
        "sweep_type": "cross_solver_per_color",
        "methodology": "uniform_cf",
        "color_wavelengths": COLOR_PEAK_WL,
        "timestamp": datetime.now(UTC).isoformat(),
        "solvers": solver_results,
    }
    save_json(json_data, output_dir, "cross_solver_per_color.json")


def sweep_angle_per_color(output_dir: str) -> None:
    """Angle-dependent per-color QE for torcwa using uniform CF.

    Sweeps CRA = 0, 15, 30 degrees at each color's peak wavelength
    with Fourier order N=5 (converged) to measure angular sensitivity per-color.

    Args:
        output_dir: Directory for JSON output.
    """
    cra_values = [0.0, 15.0, 30.0]
    N = 5  # converged torcwa order
    fo = [N, N]
    harmonics = (2 * N + 1) ** 2

    all_results = {}

    for cra in cra_values:
        print(f"\n  CRA = {cra:.0f}° (N={N}, {harmonics} harmonics):")
        cra_entry: dict = {"cra_deg": cra}
        total_time = 0.0

        for color in ["B", "G", "R"]:
            wl = COLOR_PEAK_WL[color]
            pixel_config = _pixel_config_uniform_cf(color)
            pixel_config["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = cra
            source_config = build_source_config_multi_wavelength([wl], theta_deg=cra)

            print(f"    {color} (λ={wl * 1000:.0f}nm)...", end=" ", flush=True)

            _wavelengths, _R, _T, A, _qe, elapsed = run_sim_with_qe(
                "torcwa", fo, n_lens_slices=30, grid_multiplier=3,
                source_config=source_config,
                pixel_config=pixel_config,
            )

            qe_val = float(A[0])
            cra_entry[f"QE_{color}"] = qe_val
            total_time += elapsed

            print(f"QE={qe_val:.5f} ({elapsed:.1f}s)")

        cra_entry["time_s"] = total_time
        all_results[f"cra_{cra:.0f}"] = cra_entry

    # Print angle-dependent table
    print("\n=== Per-Color QE vs CRA (torcwa N=5, uniform CF) ===")
    print(f"{'CRA':>6} | {'QE_R@620':>9} | {'QE_G@530':>9} | {'QE_B@450':>9}")
    print("-" * 45)
    for cra in cra_values:
        r = all_results[f"cra_{cra:.0f}"]
        print(f"{cra:>5.0f}° | {r['QE_R']:>9.5f} | {r['QE_G']:>9.5f} | {r['QE_B']:>9.5f}")

    # RI per color
    print("\n  Relative Illumination (QE / QE_center):")
    center = all_results["cra_0"]
    print(f"{'CRA':>6} | {'RI_R':>9} | {'RI_G':>9} | {'RI_B':>9}")
    print("-" * 45)
    for cra in cra_values:
        r = all_results[f"cra_{cra:.0f}"]
        ri_r = r["QE_R"] / center["QE_R"] if center["QE_R"] > 0 else 0
        ri_g = r["QE_G"] / center["QE_G"] if center["QE_G"] > 0 else 0
        ri_b = r["QE_B"] / center["QE_B"] if center["QE_B"] > 0 else 0
        print(f"{cra:>5.0f}° | {ri_r:>9.4f} | {ri_g:>9.4f} | {ri_b:>9.4f}")

    json_data = {
        "sweep_type": "angle_per_color",
        "methodology": "uniform_cf",
        "N": N,
        "cra_values": cra_values,
        "color_wavelengths": COLOR_PEAK_WL,
        "timestamp": datetime.now(UTC).isoformat(),
        "results": all_results,
    }
    save_json(json_data, output_dir, "angle_per_color.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="COMPASS convergence study — parametric sweeps for RCWA solver parameters.",
    )
    parser.add_argument(
        "--sweep",
        required=True,
        choices=[
            "fourier_order_grcwa",
            "fourier_order_torcwa",
            "n_lens_slices",
            "grid_resolution",
            "full_spectrum",
            "fourier_per_color_grcwa",
            "fourier_per_color_torcwa",
            "per_color_spectrum",
            "cross_solver_per_color",
            "angle_per_color",
        ],
        help="Which convergence sweep to run.",
    )
    parser.add_argument(
        "--fourier-order",
        type=int,
        default=225,
        help="Fourier order for non-Fourier sweeps (default: 225).",
    )
    parser.add_argument(
        "--n-lens-slices",
        type=int,
        default=30,
        help="Number of lens slices for non-lens-slice sweeps (default: 30).",
    )
    parser.add_argument(
        "--grid-multiplier",
        type=int,
        default=3,
        help="Grid multiplier for non-grid sweeps (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/convergence",
        help="Output directory for JSON results (default: results/convergence).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the convergence study script."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    print(f"COMPASS Convergence Study — {args.sweep}")
    print(f"Output directory: {args.output_dir}")
    t_total = time.perf_counter()

    if args.sweep == "fourier_order_grcwa":
        sweep_fourier_order_grcwa(args.output_dir)

    elif args.sweep == "fourier_order_torcwa":
        sweep_fourier_order_torcwa(args.output_dir)

    elif args.sweep == "n_lens_slices":
        sweep_n_lens_slices(args.output_dir, args.fourier_order)

    elif args.sweep == "grid_resolution":
        sweep_grid_resolution(args.output_dir, args.fourier_order, args.n_lens_slices)

    elif args.sweep == "full_spectrum":
        sweep_full_spectrum(
            args.output_dir, args.fourier_order, args.n_lens_slices, args.grid_multiplier
        )

    elif args.sweep == "fourier_per_color_grcwa":
        sweep_fourier_per_color_grcwa(args.output_dir)

    elif args.sweep == "fourier_per_color_torcwa":
        sweep_fourier_per_color_torcwa(args.output_dir)

    elif args.sweep == "per_color_spectrum":
        sweep_per_color_spectrum(args.output_dir)

    elif args.sweep == "cross_solver_per_color":
        sweep_cross_solver_per_color(args.output_dir)

    elif args.sweep == "angle_per_color":
        sweep_angle_per_color(args.output_dir)

    total_elapsed = time.perf_counter() - t_total
    print(f"\nTotal time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
