#!/usr/bin/env python3
"""Generate the COMPASS performance benchmark report.

The benchmark is intentionally lightweight and reproducible. It measures the
CPU paths that are stable in every developer environment, then adds an optional
GPU row when CUDA is available through PyTorch.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import tracemalloc
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compass.geometry.pixel_stack import PixelStack  # noqa: E402
from compass.geometry.sample_pixels import derive_parameters  # noqa: E402
from compass.materials.database import MaterialDB  # noqa: E402
from compass.solvers.base import SolverFactory  # noqa: E402

DOCS = ROOT / "docs"
PUBLIC_DIR = DOCS / "public" / "reports" / "performance-benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate performance benchmark report.")
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--core-repeats", type=int, default=5)
    parser.add_argument("--solver-repeats", type=int, default=3)
    return parser.parse_args()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        if 0 < abs(value) < 1e-4:
            return f"{value:.2e}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cells = [fmt(value).replace("|", r"\|").replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def time_operation(
    *,
    name: str,
    category: str,
    size_label: str,
    func: Callable[[], dict[str, Any]],
    repeats: int,
    warmups: int = 1,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for _ in range(max(0, warmups)):
        func()

    samples: list[float] = []
    peak_mb: list[float] = []
    last_result: dict[str, Any] = {}
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        started = perf_counter()
        last_result = func()
        elapsed = perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        samples.append(elapsed)
        peak_mb.append(peak / (1024.0 * 1024.0))

    return {
        "name": name,
        "category": category,
        "size_label": size_label,
        "status": "ok",
        "repeats": repeats,
        "warmups": warmups,
        "median_s": float(statistics.median(samples)),
        "mean_s": float(statistics.mean(samples)),
        "min_s": float(min(samples)),
        "max_s": float(max(samples)),
        "stdev_s": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        "peak_traced_mb_median": float(statistics.median(peak_mb)),
        "peak_traced_mb_max": float(max(peak_mb)),
        "samples_s": samples,
        "peak_traced_mb_samples": peak_mb,
        "result": json_safe(last_result),
        "metadata": metadata or {},
    }


def safe_time_operation(**kwargs: Any) -> dict[str, Any]:
    try:
        return time_operation(**kwargs)
    except Exception as exc:  # pragma: no cover - report must capture optional failures.
        return {
            "name": kwargs["name"],
            "category": kwargs["category"],
            "size_label": kwargs["size_label"],
            "status": "failed",
            "repeats": kwargs["repeats"],
            "warmups": kwargs.get("warmups", 1),
            "median_s": None,
            "mean_s": None,
            "min_s": None,
            "max_s": None,
            "stdev_s": None,
            "peak_traced_mb_median": None,
            "peak_traced_mb_max": None,
            "samples_s": [],
            "peak_traced_mb_samples": [],
            "result": {},
            "metadata": kwargs.get("metadata") or {},
            "error": repr(exc),
        }


def pixel_config(pattern: str = "bayer_rggb") -> dict[str, Any]:
    return {"pixel": derive_parameters("generic_bsi", pitch=1.0, cf_pattern=pattern)}


def source_config(wavelengths: np.ndarray) -> dict[str, Any]:
    return {
        "wavelength": {"mode": "list", "values": [float(wl) for wl in wavelengths]},
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "TE",
    }


def summarize_result(result: Any, device: str) -> dict[str, Any]:
    reflection = np.asarray(result.reflection, dtype=float) if result.reflection is not None else None
    transmission = (
        np.asarray(result.transmission, dtype=float) if result.transmission is not None else None
    )
    absorption = np.asarray(result.absorption, dtype=float) if result.absorption is not None else None
    if reflection is not None and transmission is not None and absorption is not None:
        energy_residual = float(np.nanmax(np.abs(reflection + transmission + absorption - 1.0)))
        mean_absorption = float(np.nanmean(absorption))
    else:
        energy_residual = None
        mean_absorption = None

    gpu_peak_mb = None
    if device.startswith("cuda"):
        try:
            import torch

            gpu_peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        except Exception:
            gpu_peak_mb = None

    return {
        "metadata_runtime_s": float(result.metadata.get("runtime_seconds", math.nan)),
        "max_energy_residual": energy_residual,
        "mean_absorption": mean_absorption,
        "gpu_peak_allocated_mb": gpu_peak_mb,
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        info["cuda_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as exc:
        info["torch_version"] = None
        info["cuda_available"] = False
        info["cuda_device_count"] = 0
        info["cuda_device_name"] = None
        info["torch_error"] = repr(exc)
    try:
        import torcwa

        info["torcwa_available"] = True
        info["torcwa_version"] = getattr(torcwa, "__version__", "unknown")
    except Exception as exc:
        info["torcwa_available"] = False
        info["torcwa_version"] = None
        info["torcwa_error"] = repr(exc)
    return info


def run_core_benchmarks(core_repeats: int) -> list[dict[str, Any]]:
    db = MaterialDB()
    wavelengths = np.linspace(0.38, 0.78, 41)
    config_2x2 = pixel_config("bayer_rggb")
    config_4x4 = pixel_config("tetracell")
    stack_2x2 = PixelStack(config_2x2)

    def construct_stack_summary(cfg: dict[str, Any]) -> dict[str, Any]:
        stack = PixelStack(cfg)
        return {"layers": len(stack.layers), "unit_cell": list(stack.unit_cell)}

    rows: list[dict[str, Any]] = []
    rows.append(
        safe_time_operation(
            name="Silicon epsilon spectrum lookup",
            category="material",
            size_label="silicon x 41 wl",
            repeats=max(core_repeats, 5),
            func=lambda: {
                "materials": len(db.list_materials()),
                "wavelengths": len(wavelengths),
                "last_shape": list(db.get_epsilon_spectrum("silicon", wavelengths).shape),
            },
        )
    )
    rows.append(
        safe_time_operation(
            name="All-material spectrum lookup",
            category="material",
            size_label=f"{len(db.list_materials())} materials x 41 wl",
            repeats=max(core_repeats, 5),
            func=lambda: {
                "materials": len(db.list_materials()),
                "wavelengths": len(wavelengths),
                "checksum": float(
                    np.real(
                        sum(
                            np.sum(db.get_epsilon_spectrum(mat, wavelengths))
                            for mat in db.list_materials()
                        )
                    )
                ),
            },
        )
    )
    for label, cfg in [("2x2 Bayer", config_2x2), ("4x4 TetraCell", config_4x4)]:
        rows.append(
            safe_time_operation(
                name="PixelStack construction",
                category="construction",
                size_label=label,
                repeats=max(core_repeats, 5),
                func=lambda cfg=cfg: construct_stack_summary(cfg),
            )
        )

    for nx in [32, 64, 128, 192]:
        rows.append(
            safe_time_operation(
                name="get_layer_slices",
                category="layer_slices",
                size_label=f"{nx}x{nx}, 8 lens slices",
                repeats=core_repeats,
                metadata={"nx": nx, "ny": nx, "grid_cells": nx * nx},
                func=lambda nx=nx: {
                    "slice_count": len(
                        stack_2x2.get_layer_slices(
                            wavelength=0.55,
                            nx=nx,
                            ny=nx,
                            n_lens_slices=8,
                        )
                    ),
                    "grid_cells": nx * nx,
                },
            )
        )

    for nx in [32, 64, 96, 128]:
        nz = 64
        rows.append(
            safe_time_operation(
                name="get_permittivity_grid",
                category="permittivity_grid",
                size_label=f"{nx}x{nx}x{nz}",
                repeats=max(3, core_repeats - 1),
                metadata={"nx": nx, "ny": nx, "nz": nz, "grid_cells": nx * nx * nz},
                func=lambda nx=nx, nz=nz: {
                    "shape": list(
                        stack_2x2.get_permittivity_grid(
                            wavelength=0.55,
                            nx=nx,
                            ny=nx,
                            nz=nz,
                        ).shape
                    ),
                    "grid_cells": nx * nx * nz,
                },
            )
        )
    return rows


def run_solver_benchmarks(solver_repeats: int, env: dict[str, Any]) -> list[dict[str, Any]]:
    stack = PixelStack(pixel_config("bayer_rggb"))
    rows: list[dict[str, Any]] = []

    def run_tmm(n_wavelengths: int) -> dict[str, Any]:
        wavelengths = np.linspace(0.40, 0.70, n_wavelengths)
        solver = SolverFactory.create(
            "tmm",
            {"name": "tmm", "type": "tmm", "params": {"polarization_average": False}},
            "cpu",
        )
        solver.setup_geometry(stack)
        solver.setup_source(source_config(wavelengths))
        result = solver.run_timed()
        return summarize_result(result, "cpu")

    for n_wavelengths in [1, 11, 31, 61, 101]:
        rows.append(
            safe_time_operation(
                name="TMM wavelength sweep",
                category="solver",
                size_label=f"{n_wavelengths} wavelengths",
                repeats=max(solver_repeats, 3),
                metadata={
                    "solver": "tmm",
                    "device": "cpu",
                    "n_wavelengths": n_wavelengths,
                },
                func=lambda n_wavelengths=n_wavelengths: run_tmm(n_wavelengths),
            )
        )

    torcwa_config = {
        "name": "torcwa",
        "type": "rcwa",
        "params": {
            "fourier_order": [1, 1],
            "dtype": "complex64",
            "n_lens_slices": 4,
            "grid_multiplier": 2,
        },
        "stability": {"precision_strategy": "mixed", "allow_tf32": False},
    }

    def run_torcwa(n_wavelengths: int, device: str) -> dict[str, Any]:
        wavelengths = np.linspace(0.45, 0.65, n_wavelengths)
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        solver = SolverFactory.create("torcwa", torcwa_config, device)
        solver.setup_geometry(stack)
        solver.setup_source(source_config(wavelengths))
        result = solver.run_timed()
        return summarize_result(result, device)

    for n_wavelengths in [1, 3, 5, 9]:
        rows.append(
            safe_time_operation(
                name="torcwa RCWA low-order sweep",
                category="solver",
                size_label=f"{n_wavelengths} wavelengths",
                repeats=solver_repeats,
                metadata={
                    "solver": "torcwa",
                    "device": "cpu",
                    "n_wavelengths": n_wavelengths,
                    "fourier_order": [1, 1],
                },
                func=lambda n_wavelengths=n_wavelengths: run_torcwa(n_wavelengths, "cpu"),
            )
        )

    if env.get("cuda_available"):
        for n_wavelengths in [1, 3, 5]:
            rows.append(
                safe_time_operation(
                    name="torcwa RCWA low-order sweep",
                    category="solver",
                    size_label=f"{n_wavelengths} wavelengths",
                    repeats=max(2, solver_repeats - 1),
                    metadata={
                        "solver": "torcwa",
                        "device": "cuda",
                        "n_wavelengths": n_wavelengths,
                        "fourier_order": [1, 1],
                    },
                    func=lambda n_wavelengths=n_wavelengths: run_torcwa(n_wavelengths, "cuda"),
                )
            )
    else:
        rows.append(
            {
                "name": "torcwa RCWA low-order sweep",
                "category": "solver",
                "size_label": "GPU",
                "status": "not_available",
                "repeats": 0,
                "warmups": 0,
                "median_s": None,
                "mean_s": None,
                "min_s": None,
                "max_s": None,
                "stdev_s": None,
                "peak_traced_mb_median": None,
                "peak_traced_mb_max": None,
                "samples_s": [],
                "peak_traced_mb_samples": [],
                "result": {},
                "metadata": {"solver": "torcwa", "device": "cuda", "n_wavelengths": None},
                "error": "CUDA is not available in this environment.",
            }
        )

    return rows


def fit_slope_ms_per_wavelength(rows: list[dict[str, Any]], solver: str, device: str) -> float | None:
    points = [
        (
            float(row["metadata"]["n_wavelengths"]),
            float(row["median_s"]),
        )
        for row in rows
        if row.get("status") == "ok"
        and row.get("metadata", {}).get("solver") == solver
        and row.get("metadata", {}).get("device") == device
        and row.get("median_s") is not None
    ]
    if len(points) < 2:
        return None
    x = np.array([item[0] for item in points], dtype=float)
    y = np.array([item[1] for item in points], dtype=float)
    slope_s = float(np.polyfit(x, y, 1)[0])
    return max(0.0, slope_s * 1000.0)


def find_solver_row(rows: list[dict[str, Any]], solver: str, device: str, n_wavelengths: int) -> dict[str, Any] | None:
    for row in rows:
        md = row.get("metadata", {})
        if (
            row.get("status") == "ok"
            and md.get("solver") == solver
            and md.get("device") == device
            and md.get("n_wavelengths") == n_wavelengths
        ):
            return row
    return None


def plot_core_runtime(rows: list[dict[str, Any]], outpath: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    labels = [f"{row['name']}\n{row['size_label']}" for row in ok_rows]
    values_ms = [float(row["median_s"]) * 1000.0 for row in ok_rows]
    colors = {
        "material": "#2563eb",
        "construction": "#16a34a",
        "layer_slices": "#f97316",
        "permittivity_grid": "#dc2626",
    }
    bar_colors = [colors.get(row["category"], "#64748b") for row in ok_rows]
    fig, ax = plt.subplots(figsize=(9.8, 7.0), constrained_layout=True)
    y = np.arange(len(ok_rows))
    ax.barh(y, values_ms, color=bar_colors, alpha=0.86)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("median runtime (ms, log scale)")
    ax.set_title("Core CPU operation runtime")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_geometry_scaling(rows: list[dict[str, Any]], outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    for ax, category, title, color in [
        (axes[0], "layer_slices", "Layer slices", "#f97316"),
        (axes[1], "permittivity_grid", "3D permittivity grid", "#dc2626"),
    ]:
        selected = [row for row in rows if row.get("status") == "ok" and row["category"] == category]
        selected.sort(key=lambda row: row["metadata"]["grid_cells"])
        x = [row["metadata"]["grid_cells"] for row in selected]
        y = [float(row["median_s"]) * 1000.0 for row in selected]
        peak = [float(row["peak_traced_mb_median"]) for row in selected]
        ax.plot(x, y, marker="o", linewidth=2.0, color=color, label="runtime")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("grid cells")
        ax.set_ylabel("median runtime (ms)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.24)
        ax2 = ax.twinx()
        ax2.plot(x, peak, marker="s", linestyle="--", color="#475569", label="traced peak")
        ax2.set_ylabel("traced peak MB")
    fig.suptitle("Geometry generation scaling")
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_solver_scaling(rows: list[dict[str, Any]], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    series = [
        ("tmm", "cpu", "TMM CPU", "#2563eb"),
        ("torcwa", "cpu", "torcwa CPU", "#dc2626"),
        ("torcwa", "cuda", "torcwa GPU", "#16a34a"),
    ]
    for solver, device, label, color in series:
        selected = [
            row
            for row in rows
            if row.get("status") == "ok"
            and row.get("metadata", {}).get("solver") == solver
            and row.get("metadata", {}).get("device") == device
        ]
        if not selected:
            continue
        selected.sort(key=lambda row: row["metadata"]["n_wavelengths"])
        x = [row["metadata"]["n_wavelengths"] for row in selected]
        y = [float(row["median_s"]) for row in selected]
        ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("wavelength count")
    ax.set_ylabel("median runtime (s, log scale)")
    ax.set_title("Solver wavelength-sweep scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_memory(rows: list[dict[str, Any]], outpath: Path) -> None:
    selected = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("peak_traced_mb_median") is not None
        and row.get("peak_traced_mb_median") > 0
    ]
    labels = [f"{row['name']}\n{row['size_label']}" for row in selected]
    values = [float(row["peak_traced_mb_median"]) for row in selected]
    fig, ax = plt.subplots(figsize=(9.8, 7.0), constrained_layout=True)
    y = np.arange(len(selected))
    ax.barh(y, values, color="#0f766e", alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median traced peak memory (MB)")
    ax.set_title("Python traced memory profile")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def env_table(env: dict[str, Any]) -> str:
    return markdown_table(
        ["Field", "Value"],
        [
            ["Platform", env.get("platform")],
            ["Python", env.get("python")],
            ["Machine", env.get("machine")],
            ["Processor", env.get("processor")],
            ["CPU count", env.get("cpu_count")],
            ["Torch", env.get("torch_version")],
            ["CUDA available", env.get("cuda_available")],
            ["CUDA devices", env.get("cuda_device_count")],
            ["CUDA device name", env.get("cuda_device_name")],
            ["torcwa available", env.get("torcwa_available")],
            ["torcwa version", env.get("torcwa_version")],
        ],
    )


def core_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        ["Category", "Benchmark", "Size", "Repeats", "Median ms", "Min ms", "Max ms", "Peak MB"],
        [
            [
                row["category"],
                row["name"],
                row["size_label"],
                row["repeats"],
                None if row.get("median_s") is None else row["median_s"] * 1000.0,
                None if row.get("min_s") is None else row["min_s"] * 1000.0,
                None if row.get("max_s") is None else row["max_s"] * 1000.0,
                row.get("peak_traced_mb_median"),
            ]
            for row in rows
        ],
    )


def solver_table(rows: list[dict[str, Any]]) -> str:
    table_rows = []
    for row in rows:
        md = row.get("metadata", {})
        n_wl = md.get("n_wavelengths")
        median_s = row.get("median_s")
        per_wl_ms = None if median_s is None or not n_wl else float(median_s) * 1000.0 / n_wl
        throughput = None if median_s is None or not n_wl else n_wl / float(median_s)
        result = row.get("result", {})
        table_rows.append(
            [
                md.get("solver", row["name"]),
                md.get("device"),
                n_wl,
                row.get("status"),
                row.get("repeats"),
                median_s,
                per_wl_ms,
                throughput,
                result.get("max_energy_residual"),
                row.get("peak_traced_mb_median"),
                result.get("gpu_peak_allocated_mb"),
            ]
        )
    return markdown_table(
        [
            "Solver",
            "Device",
            "Wavelengths",
            "Status",
            "Repeats",
            "Median s",
            "ms / wl",
            "wl / s",
            "max energy residual",
            "traced MB",
            "GPU MB",
        ],
        table_rows,
    )


def build_summary(core_rows: list[dict[str, Any]], solver_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def by_name(name: str, size_label: str | None = None) -> dict[str, Any] | None:
        for row in core_rows:
            if row["name"] == name and (size_label is None or row["size_label"] == size_label):
                return row
        return None

    tmm_31 = find_solver_row(solver_rows, "tmm", "cpu", 31)
    rcwa_5 = find_solver_row(solver_rows, "torcwa", "cpu", 5)
    tmm_slope = fit_slope_ms_per_wavelength(solver_rows, "tmm", "cpu")
    rcwa_slope = fit_slope_ms_per_wavelength(solver_rows, "torcwa", "cpu")
    tmm_per_wl = None
    rcwa_per_wl = None
    if tmm_31 and tmm_31.get("median_s") is not None:
        tmm_per_wl = float(tmm_31["median_s"]) * 1000.0 / 31.0
    if rcwa_5 and rcwa_5.get("median_s") is not None:
        rcwa_per_wl = float(rcwa_5["median_s"]) * 1000.0 / 5.0
    speed_ratio = None
    if tmm_per_wl and rcwa_per_wl:
        speed_ratio = rcwa_per_wl / tmm_per_wl
    return {
        "material_all_ms": (
            by_name("All-material spectrum lookup") or {}
        ).get("median_s"),
        "pixelstack_2x2_ms": (
            by_name("PixelStack construction", "2x2 Bayer") or {}
        ).get("median_s"),
        "pixelstack_4x4_ms": (
            by_name("PixelStack construction", "4x4 TetraCell") or {}
        ).get("median_s"),
        "tmm_31_median_s": tmm_31.get("median_s") if tmm_31 else None,
        "torcwa_5_median_s": rcwa_5.get("median_s") if rcwa_5 else None,
        "tmm_slope_ms_per_wavelength": tmm_slope,
        "torcwa_cpu_slope_ms_per_wavelength": rcwa_slope,
        "torcwa_to_tmm_per_wavelength_ratio": speed_ratio,
    }


def report_markdown(
    *,
    generated_on: str,
    env: dict[str, Any],
    core_rows: list[dict[str, Any]],
    solver_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    korean: bool = False,
) -> str:
    if not korean:
        title = "# Performance Benchmark"
        intro = (
            f"_Generated on {generated_on} from local CPU/GPU timing runs._\n\n"
            "This report publishes a lightweight performance baseline for the code paths "
            "that developers exercise most often: material lookup, PixelStack geometry "
            "generation, TMM sweeps, and low-order torcwa RCWA sweeps."
        )
        bullets = [
            f"All-material 41-wavelength lookup median: {fmt((summary['material_all_ms'] or 0.0) * 1000.0)} ms.",
            f"PixelStack construction median: {fmt((summary['pixelstack_2x2_ms'] or 0.0) * 1000.0)} ms for 2x2, {fmt((summary['pixelstack_4x4_ms'] or 0.0) * 1000.0)} ms for 4x4.",
            f"TMM 31-wavelength sweep median: {fmt(summary['tmm_31_median_s'])} s; fitted cost {fmt(summary['tmm_slope_ms_per_wavelength'])} ms/wavelength.",
            f"torcwa CPU 5-wavelength low-order sweep median: {fmt(summary['torcwa_5_median_s'])} s; fitted cost {fmt(summary['torcwa_cpu_slope_ms_per_wavelength'])} ms/wavelength.",
        ]
        gpu_note = (
            "CUDA was available and GPU rows are included."
            if env.get("cuda_available")
            else "CUDA was not available in this environment, so GPU rows are recorded as not available."
        )
        scope = (
            "::: warning Scope\n"
            "These are local timing numbers, not universal performance guarantees. The RCWA "
            "case is a low-order smoke benchmark, and traced memory excludes some native "
            "library allocations such as PyTorch/CUDA allocator internals.\n"
            ":::"
        )
        env_heading = "## Environment"
        core_heading = "## Core CPU Operations"
        geom_heading = "## Geometry Scaling"
        solver_heading = "## Solver Sweep Cost"
        interp_heading = "## Interpretation"
        interp = [
            "- TMM remains the fast path for stack screening and wavelength-dense BARL sweeps.",
            "- Low-order torcwa CPU runtime scales close to linearly with wavelength count, so "
            "larger reports should budget by wavelength first, then Fourier order.",
            "- Geometry generation is cheap enough for repeated report generation at the tested "
            "resolutions, but 3D permittivity grids dominate traced memory.",
            "- The GPU row is intentionally conditional; machines without CUDA still generate a "
            "complete CPU report.",
        ]
        regen_heading = "## Regeneration"
        regen = "```powershell\nuv run python scripts\\generate_performance_benchmark_report.py\n```"
        metrics = (
            "Generated metrics are stored at "
            "`docs/public/reports/performance-benchmark/performance_benchmark_metrics.json`."
        )
    else:
        title = "# Performance Benchmark"
        intro = (
            f"_생성일: {generated_on}. 로컬 CPU/GPU timing run에서 생성._\n\n"
            "이 리포트는 개발자가 자주 실행하는 code path의 가벼운 성능 기준선을 공개한다: "
            "material lookup, PixelStack geometry 생성, TMM sweep, 저차 torcwa RCWA sweep."
        )
        bullets = [
            f"전체 material 41-wavelength lookup median: {fmt((summary['material_all_ms'] or 0.0) * 1000.0)} ms.",
            f"PixelStack construction median: 2x2 {fmt((summary['pixelstack_2x2_ms'] or 0.0) * 1000.0)} ms, 4x4 {fmt((summary['pixelstack_4x4_ms'] or 0.0) * 1000.0)} ms.",
            f"TMM 31-wavelength sweep median: {fmt(summary['tmm_31_median_s'])} s; fitted cost {fmt(summary['tmm_slope_ms_per_wavelength'])} ms/wavelength.",
            f"torcwa CPU 5-wavelength low-order sweep median: {fmt(summary['torcwa_5_median_s'])} s; fitted cost {fmt(summary['torcwa_cpu_slope_ms_per_wavelength'])} ms/wavelength.",
        ]
        gpu_note = (
            "CUDA를 사용할 수 있어 GPU row를 포함했다."
            if env.get("cuda_available")
            else "이 환경에서는 CUDA를 사용할 수 없어 GPU row는 not available로 기록했다."
        )
        scope = (
            "::: warning Scope\n"
            "이 값은 로컬 timing 숫자이며 모든 환경의 성능 보장이 아니다. RCWA case는 "
            "low-order smoke benchmark이고, traced memory는 PyTorch/CUDA allocator 내부 같은 "
            "일부 native allocation을 제외할 수 있다.\n"
            ":::"
        )
        env_heading = "## Environment"
        core_heading = "## Core CPU Operations"
        geom_heading = "## Geometry Scaling"
        solver_heading = "## Solver Sweep Cost"
        interp_heading = "## Interpretation"
        interp = [
            "- TMM은 stack screening과 wavelength가 많은 BARL sweep의 빠른 경로다.",
            "- 저차 torcwa CPU runtime은 wavelength count에 거의 선형으로 증가하므로, 큰 "
            "리포트는 wavelength 수를 먼저 예산화하고 그 다음 Fourier order를 조정한다.",
            "- 테스트한 해상도에서 geometry generation은 반복 리포트 생성에 충분히 가볍지만, "
            "3D permittivity grid가 traced memory를 지배한다.",
            "- GPU row는 조건부다. CUDA가 없는 장비도 완전한 CPU 리포트를 생성한다.",
        ]
        regen_heading = "## Regeneration"
        regen = "```powershell\nuv run python scripts\\generate_performance_benchmark_report.py\n```"
        metrics = (
            "생성된 metric은 "
            "`docs/public/reports/performance-benchmark/performance_benchmark_metrics.json`에 저장된다."
        )

    lines = [
        "---",
        "outline: deep",
        "---",
        "",
        title,
        "",
        intro,
        "",
        "## Executive summary" if not korean else "## 요약",
        "",
        *[f"- {item}" for item in bullets],
        f"- {gpu_note}",
        "",
        scope,
        "",
        env_heading,
        "",
        env_table(env),
        "",
        core_heading,
        "",
        "![Core operation runtime](/reports/performance-benchmark/01_core_operation_runtime.png)",
        "",
        core_table(core_rows),
        "",
        geom_heading,
        "",
        "![Geometry scaling](/reports/performance-benchmark/02_geometry_scaling.png)",
        "",
        "![Memory profile](/reports/performance-benchmark/03_memory_profile.png)",
        "",
        solver_heading,
        "",
        "![Solver wavelength scaling](/reports/performance-benchmark/04_solver_wavelength_scaling.png)",
        "",
        solver_table(solver_rows),
        "",
        interp_heading,
        "",
        *interp,
        "",
        regen_heading,
        "",
        regen,
        "",
        metrics,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    docs = args.docs
    public_dir = docs / "public" / "reports" / "performance-benchmark"
    reports = docs / "reports"
    reports_ko = docs / "ko" / "reports"
    public_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    reports_ko.mkdir(parents=True, exist_ok=True)

    env = environment_info()
    core_rows = run_core_benchmarks(args.core_repeats)
    solver_rows = run_solver_benchmarks(args.solver_repeats, env)
    summary = build_summary(core_rows, solver_rows)

    plot_core_runtime(core_rows, public_dir / "01_core_operation_runtime.png")
    plot_geometry_scaling(core_rows, public_dir / "02_geometry_scaling.png")
    plot_memory(core_rows + solver_rows, public_dir / "03_memory_profile.png")
    plot_solver_scaling(solver_rows, public_dir / "04_solver_wavelength_scaling.png")

    en = report_markdown(
        generated_on=args.date,
        env=env,
        core_rows=core_rows,
        solver_rows=solver_rows,
        summary=summary,
    )
    ko = report_markdown(
        generated_on=args.date,
        env=env,
        core_rows=core_rows,
        solver_rows=solver_rows,
        summary=summary,
        korean=True,
    )
    (reports / "performance-benchmark.md").write_text(en, encoding="utf-8")
    (reports_ko / "performance-benchmark.md").write_text(ko, encoding="utf-8")

    metrics = {
        "benchmark": {
            "name": "performance-benchmark",
            "generated_on": args.date,
            "core_repeats": args.core_repeats,
            "solver_repeats": args.solver_repeats,
        },
        "environment": env,
        "summary": summary,
        "core_operations": core_rows,
        "solver_sweeps": solver_rows,
    }
    (public_dir / "performance_benchmark_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote performance benchmark report and assets to {public_dir}")


if __name__ == "__main__":
    main()
