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


def run_single_sim(
    solver_name: str,
    fourier_order: list[int],
    n_lens_slices: int,
    grid_multiplier: int,
    source_config: dict,
) -> tuple[float, float, float, float]:
    """Run a single simulation and return R, T, A, elapsed time.

    Args:
        solver_name: Solver name ("grcwa" or "torcwa").
        fourier_order: Fourier order as [N, N] or [nG, nG].
        n_lens_slices: Number of microlens staircase slices.
        grid_multiplier: Grid resolution multiplier.
        source_config: Source configuration dictionary.

    Returns:
        Tuple of (R, T, A, elapsed_seconds).
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

    total_elapsed = time.perf_counter() - t_total
    print(f"\nTotal time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
