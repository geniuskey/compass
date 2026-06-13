#!/usr/bin/env python3
"""Generate the CRA cone illumination sweep report.

This report exercises two complementary paths:

* TMM cone averaging for cheap angular-integration convergence and
  CRA/F-number response maps.
* A small torcwa PixelStack smoke sweep to verify that cone illumination,
  CRA, and microlens shift settings pass through the real patterned solver
  adapter path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compass.geometry.sample_pixels import derive_parameters  # noqa: E402
from compass.runners.cone_runner import ConeIlluminationRunner  # noqa: E402
from compass.sources.cone_illumination import ConeIllumination  # noqa: E402

DOCS = ROOT / "docs"
PUBLIC_DIR = DOCS / "public" / "reports" / "cra-cone"
WAVELENGTHS = np.array([0.45, 0.55, 0.65], dtype=float)
CRA_VALUES = [0.0, 10.0, 20.0, 30.0]
F_NUMBERS = [1.4, 2.0, 2.8, 4.0]
SAMPLING_METHODS = ["fibonacci", "rings", "halton", "hammersley", "grid"]
CONVERGENCE_POINTS = [5, 13, 25, 49]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CRA cone illumination report.")
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--date", default=date.today().isoformat())
    return parser.parse_args()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
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
        cells = [fmt(item).replace("|", r"\|").replace("\n", "<br>") for item in row]
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


def source_config(
    wavelengths: np.ndarray,
    cra_deg: float,
    f_number: float,
    n_points: int,
    sampling: str,
    polarization: str = "unpolarized",
) -> dict[str, Any]:
    return {
        "wavelength": {
            "mode": "list",
            "values": [float(wl) for wl in wavelengths],
        },
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": polarization,
        "cone": {
            "cra_deg": float(cra_deg),
            "f_number": float(f_number),
            "sampling": {
                "type": sampling,
                "n_points": int(n_points),
            },
            "weighting": "cosine",
        },
    }


def pixel_config_for_shift(cra_deg: float, shift_mode: str) -> dict[str, Any]:
    pixel = copy.deepcopy(derive_parameters("generic_bsi", pitch=1.0))
    shift = pixel["layers"]["microlens"].setdefault("shift", {})
    if shift_mode == "auto_cra":
        shift["mode"] = "auto_cra"
        shift["cra_deg"] = float(cra_deg)
    else:
        shift["mode"] = "none"
        shift["cra_deg"] = 0.0
    return pixel


def solver_config(name: str) -> dict[str, Any]:
    if name == "tmm":
        return {
            "name": "tmm",
            "type": "tmm",
            "params": {
                "polarization_average": True,
            },
        }
    if name == "torcwa":
        return {
            "name": "torcwa",
            "type": "rcwa",
            "params": {
                "fourier_order": [1, 1],
                "dtype": "complex64",
                "n_lens_slices": 4,
                "grid_multiplier": 2,
            },
            "stability": {
                "precision_strategy": "mixed",
                "allow_tf32": False,
            },
        }
    raise ValueError(f"Unsupported solver: {name}")


def run_cone(
    solver_name: str,
    source: dict[str, Any],
    pixel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if pixel is None:
        pixel = derive_parameters("generic_bsi", pitch=1.0)
    cfg = {
        "pixel": pixel,
        "compute": {"backend": "cpu"},
        "solver": solver_config(solver_name),
        "source": source,
    }
    started = time.perf_counter()
    try:
        result = ConeIlluminationRunner.run(cfg)
        runtime_s = time.perf_counter() - started
        return {
            "ok": True,
            "runtime_s": runtime_s,
            "wavelength_um": np.asarray(result.wavelengths, dtype=float),
            "reflection": np.asarray(result.reflection, dtype=float),
            "transmission": np.asarray(result.transmission, dtype=float),
            "absorption": np.asarray(result.absorption, dtype=float),
            "qe_per_pixel": {
                key: np.asarray(value, dtype=float)
                for key, value in result.qe_per_pixel.items()
            },
            "metadata": dict(result.metadata),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - report captures solver failures.
        runtime_s = time.perf_counter() - started
        n_wl = len(source.get("wavelength", {}).get("values", WAVELENGTHS))
        return {
            "ok": False,
            "runtime_s": runtime_s,
            "wavelength_um": np.full(n_wl, np.nan),
            "reflection": np.full(n_wl, np.nan),
            "transmission": np.full(n_wl, np.nan),
            "absorption": np.full(n_wl, np.nan),
            "qe_per_pixel": {},
            "metadata": {},
            "error": repr(exc),
        }


def mean_qe(qe_per_pixel: dict[str, np.ndarray]) -> np.ndarray:
    if not qe_per_pixel:
        return np.array([])
    return np.mean(np.vstack(list(qe_per_pixel.values())), axis=0)


def channel_qe(qe_per_pixel: dict[str, np.ndarray], channel: str) -> float | None:
    arrays = [arr for name, arr in qe_per_pixel.items() if name.startswith(channel)]
    if not arrays:
        return None
    return float(np.mean(np.vstack(arrays)))


def energy_residual(result: dict[str, Any]) -> float:
    rta = result["reflection"] + result["transmission"] + result["absorption"]
    if not np.all(np.isfinite(rta)):
        return math.nan
    return float(np.max(np.abs(rta - 1.0)))


def cone_points(cra_deg: float, f_number: float, n_points: int, sampling: str) -> np.ndarray:
    cone = ConeIllumination(
        cra_deg=cra_deg,
        f_number=f_number,
        n_points=n_points,
        sampling=sampling,
        weighting="cosine",
    )
    points = np.array(cone.get_sampling_points(), dtype=float)
    theta = np.deg2rad(points[:, 0])
    phi = np.deg2rad(points[:, 1])
    sx = np.sin(theta) * np.cos(phi)
    sy = np.sin(theta) * np.sin(phi)
    return np.column_stack([sx, sy, points[:, 2]])


def generate_sampling_maps(outpath: Path) -> None:
    scenarios = [
        (0.0, 2.0, "center"),
        (15.0, 2.0, "edge"),
        (30.0, 2.0, "corner"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), constrained_layout=True)
    for ax, (cra, fnum, label) in zip(axes, scenarios):
        pts = cone_points(cra, fnum, 37, "hammersley")
        sizes = 1200.0 * pts[:, 2] / np.max(pts[:, 2])
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=sizes,
            c=pts[:, 2],
            cmap="viridis",
            edgecolors="#111827",
            linewidths=0.25,
        )
        ax.scatter([math.sin(math.radians(cra))], [0.0], c="#dc2626", s=46, marker="x")
        ax.axhline(0.0, color="#94a3b8", linewidth=0.8)
        ax.axvline(0.0, color="#94a3b8", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.2, 0.72)
        ax.set_ylim(-0.45, 0.45)
        ax.set_title(f"{label}: CRA {cra:.0f} deg, F/{fnum:.1f}")
        ax.set_xlabel("sin(theta) cos(phi)")
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("sin(theta) sin(phi)")
    fig.suptitle("Cone sampling directions, weighted Hammersley 37-point set", fontsize=14)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def tmm_convergence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference = run_cone(
        "tmm",
        source_config(WAVELENGTHS, cra_deg=20.0, f_number=2.0, n_points=181, sampling="hammersley"),
    )
    rows: list[dict[str, Any]] = []
    for method in SAMPLING_METHODS:
        for n_points in CONVERGENCE_POINTS:
            result = run_cone(
                "tmm",
                source_config(
                    WAVELENGTHS,
                    cra_deg=20.0,
                    f_number=2.0,
                    n_points=n_points,
                    sampling=method,
                ),
            )
            if result["ok"] and reference["ok"]:
                max_delta = float(np.max(np.abs(result["absorption"] - reference["absorption"])))
                delta_550 = float(
                    abs(
                        result["absorption"][1]
                        - reference["absorption"][1]
                    )
                )
            else:
                max_delta = math.nan
                delta_550 = math.nan
            rows.append(
                {
                    "sampling": method,
                    "n_points": n_points,
                    "max_absorption_delta_vs_ref": max_delta,
                    "absorption_delta_550": delta_550,
                    "runtime_s": result["runtime_s"],
                    "absorption": result["absorption"],
                    "error": result["error"],
                }
            )
    return rows, reference


def plot_convergence(rows: list[dict[str, Any]], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    markers = {
        "fibonacci": "o",
        "rings": "s",
        "halton": "^",
        "hammersley": "D",
        "grid": "x",
    }
    for method in SAMPLING_METHODS:
        subset = [row for row in rows if row["sampling"] == method]
        xs = [row["n_points"] for row in subset]
        ys = [row["max_absorption_delta_vs_ref"] for row in subset]
        ax.plot(xs, ys, marker=markers[method], linewidth=1.8, label=method)
    ax.set_yscale("log")
    ax.set_xlabel("cone sample count")
    ax.set_ylabel("max |A - A_ref| over 450/550/650 nm")
    ax.set_title("TMM cone integration convergence at CRA 20 deg, F/2.0")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def tmm_cra_fnumber_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cra in CRA_VALUES:
        for f_number in F_NUMBERS:
            result = run_cone(
                "tmm",
                source_config(
                    WAVELENGTHS,
                    cra_deg=cra,
                    f_number=f_number,
                    n_points=49,
                    sampling="hammersley",
                ),
            )
            rows.append(
                {
                    "cra_deg": cra,
                    "f_number": f_number,
                    "runtime_s": result["runtime_s"],
                    "reflection": result["reflection"],
                    "transmission": result["transmission"],
                    "absorption": result["absorption"],
                    "energy_residual": energy_residual(result),
                    "error": result["error"],
                }
            )
    return rows


def plot_tmm_grid(rows: list[dict[str, Any]], outpath: Path) -> None:
    absorption = np.full((len(CRA_VALUES), len(F_NUMBERS)), np.nan)
    reflection = np.full_like(absorption, np.nan)
    for row in rows:
        i = CRA_VALUES.index(row["cra_deg"])
        j = F_NUMBERS.index(row["f_number"])
        absorption[i, j] = row["absorption"][1]
        reflection[i, j] = row["reflection"][1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, data, title in [
        (axes[0], absorption, "TMM cone absorption at 550 nm"),
        (axes[1], reflection, "TMM cone reflection at 550 nm"),
    ]:
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="magma")
        ax.set_xticks(range(len(F_NUMBERS)), [f"F/{f:g}" for f in F_NUMBERS])
        ax.set_yticks(range(len(CRA_VALUES)), [f"{cra:.0f}" for cra in CRA_VALUES])
        ax.set_xlabel("F-number")
        ax.set_ylabel("CRA (deg)")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, fmt(float(data[i, j]), 3), ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, shrink=0.86)
    fig.suptitle("Finite-aperture planar-stack response, Hammersley 49-point cone", fontsize=14)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def torcwa_cra_smoke() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shift_mode in ["none", "auto_cra"]:
        for cra in CRA_VALUES:
            source = source_config(
                np.array([0.55], dtype=float),
                cra_deg=cra,
                f_number=2.0,
                n_points=5,
                sampling="fibonacci",
                polarization="TE",
            )
            result = run_cone(
                "torcwa",
                source,
                pixel=pixel_config_for_shift(cra, shift_mode),
            )
            qe_mean = mean_qe(result["qe_per_pixel"])
            rows.append(
                {
                    "cra_deg": cra,
                    "shift_mode": shift_mode,
                    "ok": result["ok"],
                    "runtime_s": result["runtime_s"],
                    "reflection": float(result["reflection"][0]),
                    "transmission": float(result["transmission"][0]),
                    "absorption": float(result["absorption"][0]),
                    "mean_qe": float(qe_mean[0]) if qe_mean.size else math.nan,
                    "qe_R": channel_qe(result["qe_per_pixel"], "R"),
                    "qe_G": channel_qe(result["qe_per_pixel"], "G"),
                    "qe_B": channel_qe(result["qe_per_pixel"], "B"),
                    "energy_residual": energy_residual(result),
                    "error": result["error"],
                }
            )
    return rows


def plot_torcwa_smoke(rows: list[dict[str, Any]], outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    colors = {"none": "#dc2626", "auto_cra": "#2563eb"}
    labels = {"none": "no microlens shift", "auto_cra": "auto CRA shift"}

    for shift_mode in ["none", "auto_cra"]:
        subset = [row for row in rows if row["shift_mode"] == shift_mode]
        axes[0].plot(
            [row["cra_deg"] for row in subset],
            [row["mean_qe"] for row in subset],
            marker="o",
            linewidth=2.0,
            color=colors[shift_mode],
            label=labels[shift_mode],
        )
    auto = {row["cra_deg"]: row for row in rows if row["shift_mode"] == "auto_cra"}
    none = {row["cra_deg"]: row for row in rows if row["shift_mode"] == "none"}
    improvements = [
        auto[cra]["mean_qe"] - none[cra]["mean_qe"]
        for cra in CRA_VALUES
    ]
    axes[1].bar(CRA_VALUES, improvements, width=4.5, color="#16a34a", alpha=0.82)

    axes[0].set_xlabel("CRA (deg)")
    axes[0].set_ylabel("mean QE proxy at 550 nm")
    axes[0].set_title("Low-order torcwa cone smoke")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="#111827", linewidth=0.9)
    axes[1].set_xlabel("CRA (deg)")
    axes[1].set_ylabel("auto shift - no shift")
    axes[1].set_title("Microlens shift delta")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Patterned PixelStack cone path: F/2.0, 5 samples, TE, 550 nm", fontsize=14)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def convergence_table(rows: list[dict[str, Any]]) -> str:
    table_rows: list[list[Any]] = []
    for method in SAMPLING_METHODS:
        subset = {row["n_points"]: row for row in rows if row["sampling"] == method}
        table_rows.append(
            [
                method,
                subset[5]["max_absorption_delta_vs_ref"],
                subset[13]["max_absorption_delta_vs_ref"],
                subset[25]["max_absorption_delta_vs_ref"],
                subset[49]["max_absorption_delta_vs_ref"],
            ]
        )
    return markdown_table(["sampling", "5 pts", "13 pts", "25 pts", "49 pts"], table_rows)


def torcwa_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        [
            "CRA",
            "shift",
            "R@550",
            "T@550",
            "A@550",
            "mean QE@550",
            "QE_R",
            "QE_G",
            "QE_B",
            "energy residual",
            "runtime s",
            "error",
        ],
        [
            [
                row["cra_deg"],
                row["shift_mode"],
                row["reflection"],
                row["transmission"],
                row["absorption"],
                row["mean_qe"],
                row["qe_R"],
                row["qe_G"],
                row["qe_B"],
                row["energy_residual"],
                row["runtime_s"],
                row["error"] or "-",
            ]
            for row in rows
        ],
    )


def tmm_summary_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    summary: list[list[Any]] = []
    for cra in CRA_VALUES:
        subset = [row for row in rows if row["cra_deg"] == cra]
        best_absorption = max(float(row["absorption"][1]) for row in subset)
        worst_absorption = min(float(row["absorption"][1]) for row in subset)
        summary.append([cra, worst_absorption, best_absorption, best_absorption - worst_absorption])
    return summary


def write_reports(
    docs_root: Path,
    generated_on: str,
    conv_rows: list[dict[str, Any]],
    tmm_rows: list[dict[str, Any]],
    torcwa_rows: list[dict[str, Any]],
) -> None:
    reports = docs_root / "reports"
    reports_ko = docs_root / "ko" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    reports_ko.mkdir(parents=True, exist_ok=True)

    finite_conv = [
        row["max_absorption_delta_vs_ref"]
        for row in conv_rows
        if math.isfinite(row["max_absorption_delta_vs_ref"])
    ]
    best_49 = min(
        row["max_absorption_delta_vs_ref"]
        for row in conv_rows
        if row["n_points"] == 49 and math.isfinite(row["max_absorption_delta_vs_ref"])
    )
    torcwa_auto = [row for row in torcwa_rows if row["shift_mode"] == "auto_cra"]
    torcwa_none = [row for row in torcwa_rows if row["shift_mode"] == "none"]
    auto_by_cra = {row["cra_deg"]: row for row in torcwa_auto}
    none_by_cra = {row["cra_deg"]: row for row in torcwa_none}
    shift_deltas = [
        auto_by_cra[cra]["mean_qe"] - none_by_cra[cra]["mean_qe"]
        for cra in CRA_VALUES
    ]
    max_shift_delta = max(shift_deltas)
    min_shift_delta = min(shift_deltas)
    conv_max = max(finite_conv) if finite_conv else math.nan

    common_assets = [
        "![Cone sampling maps](/reports/cra-cone/01_cone_sampling_maps.png)",
        "![TMM cone convergence](/reports/cra-cone/02_tmm_cone_convergence.png)",
        "![TMM CRA F-number response](/reports/cra-cone/03_tmm_cra_fnumber_response.png)",
        "![torcwa CRA shift smoke](/reports/cra-cone/04_torcwa_cra_shift_smoke.png)",
    ]

    en = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# CRA Cone Illumination Sweep",
            "",
            f"_Generated on {generated_on} from `ConeIlluminationRunner` and the real PixelStack path._",
            "",
            "This report validates the cone-illumination workflow before using it for "
            "larger edge-of-sensor studies. It separates cheap planar-stack integration "
            "checks from a low-order patterned `torcwa` smoke run.",
            "",
            "## Executive summary",
            "",
            f"- TMM cone integration was swept over {len(SAMPLING_METHODS)} sampling methods "
            f"and {len(CONVERGENCE_POINTS)} sample counts; the worst sampled max |A-A_ref| "
            f"was {fmt(conv_max)}, and the best 49-point result was {fmt(best_49)}.",
            "- CRA/F-number maps were generated for CRA 0, 10, 20, and 30 deg across "
            "F/1.4, F/2.0, F/2.8, and F/4.0 using a 49-point Hammersley cone.",
            f"- The patterned `torcwa` smoke run used F/2.0, five angular samples, TE "
            f"polarization, and 550 nm. The auto-shift mean-QE delta ranged from "
            f"{fmt(min_shift_delta)} to {fmt(max_shift_delta)} in this low-order check.",
            "",
            "::: warning Scope",
            "The `torcwa` section is a low-order path check, not a converged edge-pixel "
            "design result. Use it to verify that CRA and microlens shift are wired into "
            "the solver path, then increase Fourier order and cone samples for production.",
            ":::",
            "",
            "## Cone Sampling Maps",
            "",
            common_assets[0],
            "",
            "The red cross marks the chief ray. Marker area follows the normalized cone "
            "integration weight.",
            "",
            "## TMM Integration Convergence",
            "",
            common_assets[1],
            "",
            "Reference: TMM, CRA 20 deg, F/2.0, 181-point Hammersley cone, wavelengths "
            "450/550/650 nm.",
            "",
            convergence_table(conv_rows),
            "",
            "## CRA and F-number Response",
            "",
            common_assets[2],
            "",
            markdown_table(
                ["CRA", "min A@550 over F/#", "max A@550 over F/#", "range"],
                tmm_summary_rows(tmm_rows),
            ),
            "",
            "## Patterned torcwa Smoke",
            "",
            common_assets[3],
            "",
            torcwa_table(torcwa_rows),
            "",
            "## Interpretation",
            "",
            "- In this symmetric planar TMM gate, grid and Fibonacci both converge tightly "
            "by 49 samples. For patterned RCWA workflows, low-discrepancy sampling remains "
            "the safer default because it avoids structured angular bias.",
            "- TMM isolates the angular-integration behavior from lateral pixel geometry. "
            "That makes it useful for convergence gates, but it does not model microlens "
            "focus or crosstalk.",
            "- The `torcwa` smoke run exercises the actual patterned PixelStack path. The "
            "numbers are intentionally labeled as QE proxies because the Fourier order and "
            "sample count are deliberately small.",
            "",
            "## Regeneration",
            "",
            "```powershell",
            "uv run python scripts\\generate_cra_cone_report.py",
            "```",
            "",
            "Generated metrics are stored at "
            "`docs/public/reports/cra-cone/cra_cone_metrics.json`.",
            "",
        ]
    )

    ko = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# CRA Cone Illumination Sweep 리포트",
            "",
            f"_생성일: {generated_on}. `ConeIlluminationRunner`와 실제 PixelStack 경로에서 생성._",
            "",
            "이 리포트는 더 큰 sensor edge 연구에 쓰기 전에 cone illumination workflow를 "
            "검증한다. 빠른 planar-stack integration check와 낮은 order의 patterned "
            "`torcwa` smoke run을 분리했다.",
            "",
            "## 요약",
            "",
            f"- TMM cone integration은 {len(SAMPLING_METHODS)}개 sampling method와 "
            f"{len(CONVERGENCE_POINTS)}개 sample count로 sweep했다. sampled max |A-A_ref| "
            f"최대값은 {fmt(conv_max)}, 49-point 최선 결과는 {fmt(best_49)}다.",
            "- CRA/F-number map은 CRA 0, 10, 20, 30 deg와 F/1.4, F/2.0, F/2.8, "
            "F/4.0에 대해 49-point Hammersley cone으로 생성했다.",
            f"- patterned `torcwa` smoke run은 F/2.0, angular sample 5개, TE, "
            f"550 nm 조건이다. 이 low-order check에서 auto-shift mean-QE delta는 "
            f"{fmt(min_shift_delta)}부터 {fmt(max_shift_delta)}까지였다.",
            "",
            "::: warning 범위",
            "`torcwa` 섹션은 low-order path check이며, converged edge-pixel 설계 결과가 "
            "아니다. CRA와 microlens shift가 solver path에 연결되는지 확인한 뒤, "
            "production에서는 Fourier order와 cone sample을 늘려야 한다.",
            ":::",
            "",
            "## Cone sampling map",
            "",
            common_assets[0],
            "",
            "빨간 십자는 chief ray다. marker 면적은 normalized cone integration weight를 따른다.",
            "",
            "## TMM integration convergence",
            "",
            common_assets[1],
            "",
            "Reference: TMM, CRA 20 deg, F/2.0, 181-point Hammersley cone, "
            "wavelength 450/550/650 nm.",
            "",
            convergence_table(conv_rows),
            "",
            "## CRA 및 F-number 응답",
            "",
            common_assets[2],
            "",
            markdown_table(
                ["CRA", "min A@550 over F/#", "max A@550 over F/#", "range"],
                tmm_summary_rows(tmm_rows),
            ),
            "",
            "## Patterned torcwa smoke",
            "",
            common_assets[3],
            "",
            torcwa_table(torcwa_rows),
            "",
            "## 해석",
            "",
            "- 이 대칭 planar TMM gate에서는 grid와 Fibonacci가 49 sample에서 모두 잘 "
            "수렴한다. Patterned RCWA workflow에서는 structured angular bias를 피하기 위해 "
            "low-discrepancy sampling을 기본값으로 두는 편이 안전하다.",
            "- TMM은 angular integration behavior를 lateral pixel geometry에서 분리한다. "
            "따라서 convergence gate에는 유용하지만 microlens focus나 crosstalk는 모델링하지 않는다.",
            "- `torcwa` smoke run은 실제 patterned PixelStack 경로를 실행한다. Fourier order와 "
            "sample count가 의도적으로 낮기 때문에 숫자는 QE proxy로 표시한다.",
            "",
            "## 재생성",
            "",
            "```powershell",
            "uv run python scripts\\generate_cra_cone_report.py",
            "```",
            "",
            "생성 metric은 `docs/public/reports/cra-cone/cra_cone_metrics.json`에 저장된다.",
            "",
        ]
    )

    (reports / "cra-cone-illumination-sweep.md").write_text(en, encoding="utf-8")
    (reports_ko / "cra-cone-illumination-sweep.md").write_text(ko, encoding="utf-8")


def main() -> None:
    args = parse_args()
    docs_root = args.docs.resolve()
    public_dir = docs_root / "public" / "reports" / "cra-cone"
    public_dir.mkdir(parents=True, exist_ok=True)

    generate_sampling_maps(public_dir / "01_cone_sampling_maps.png")
    conv_rows, reference = tmm_convergence()
    plot_convergence(conv_rows, public_dir / "02_tmm_cone_convergence.png")
    tmm_rows = tmm_cra_fnumber_grid()
    plot_tmm_grid(tmm_rows, public_dir / "03_tmm_cra_fnumber_response.png")
    torcwa_rows = torcwa_cra_smoke()
    plot_torcwa_smoke(torcwa_rows, public_dir / "04_torcwa_cra_shift_smoke.png")

    metrics = {
        "generated_on": args.date,
        "benchmark": {
            "pixel": "generic_bsi",
            "pitch_um": 1.0,
            "wavelength_um": WAVELENGTHS,
            "cra_deg": CRA_VALUES,
            "f_number": F_NUMBERS,
            "sampling_methods": SAMPLING_METHODS,
            "convergence_points": CONVERGENCE_POINTS,
        },
        "tmm_reference": reference,
        "tmm_convergence": conv_rows,
        "tmm_cra_fnumber_grid": tmm_rows,
        "torcwa_cra_smoke": torcwa_rows,
    }
    (public_dir / "cra_cone_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_reports(docs_root, args.date, conv_rows, tmm_rows, torcwa_rows)
    print(f"Wrote CRA cone illumination report and assets to {public_dir}")


if __name__ == "__main__":
    main()
