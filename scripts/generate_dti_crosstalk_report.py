#!/usr/bin/env python3
"""Generate the DTI crosstalk benchmark report.

The report keeps regeneration lightweight by separating:

* PixelStack geometry sweeps for FDTI/BDTI width and depth.
* A representative scalar FDTD crosstalk matrix snapshot from the convergence
  ladder, embedded here as fixed benchmark evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compass.geometry.pixel_stack import PixelStack  # noqa: E402
from compass.geometry.sample_pixels import derive_parameters  # noqa: E402
from compass.materials.database import MaterialDB  # noqa: E402

DOCS = ROOT / "docs"
PUBLIC_DIR = DOCS / "public" / "reports" / "dti-crosstalk"
WAVELENGTH_UM = 0.55
WIDTH_VALUES = [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
DEPTH_VALUES = [0.0, 0.3, 0.6, 1.2, 1.8, 2.4, 2.9]
PIXEL_LABELS = ["R_0_0", "G_0_1", "G_1_0", "B_1_1"]


@dataclass(frozen=True)
class BenchmarkSnapshot:
    mode: str
    mean_self_collection_fraction: float
    max_neighbor_crosstalk_fraction: float
    mean_total_pd_signal: float
    max_energy_tail_relative_change: float
    matrix: list[list[float]]


PIXEL_BENCHMARK = [
    BenchmarkSnapshot(
        mode="fdti",
        mean_self_collection_fraction=0.5830873154469457,
        max_neighbor_crosstalk_fraction=0.25822417836126677,
        mean_total_pd_signal=15320.354485369142,
        max_energy_tail_relative_change=0.04709843318759477,
        matrix=[
            [0.7931746082703969, 0.10057476339386437, 0.10057476339386437, 0.005675864941874426],
            [0.25822417836126677, 0.470175371001563, 0.20549461475388575, 0.0661058358832845],
            [0.25822417836126677, 0.20549461475388575, 0.470175371001563, 0.06610583588328452],
            [0.060185736245605345, 0.1704951761200673, 0.1704951761200673, 0.5988239115142601],
        ],
    ),
    BenchmarkSnapshot(
        mode="bdti_0p6um",
        mean_self_collection_fraction=0.5827451946048239,
        max_neighbor_crosstalk_fraction=0.2609124864402936,
        mean_total_pd_signal=15249.179770690309,
        max_energy_tail_relative_change=0.04696282874027519,
        matrix=[
            [0.7939483152159363, 0.10014745639953135, 0.10014745639953135, 0.0057567719850010095],
            [0.2609124864402936, 0.4681446981797666, 0.20343591261316155, 0.06750690276677819],
            [0.2609124864402935, 0.20343591261316155, 0.4681446981797666, 0.06750690276677819],
            [0.06127466674783031, 0.16899113320417172, 0.16899113320417175, 0.6007430668438262],
        ],
    ),
]

PERIODIC_TRENCH_SUMMARY = {
    "fdti": {
        "max_abs_error_R": 0.026557762271347846,
        "max_abs_error_T": 0.02698011935277761,
        "max_abs_error_A": 0.027838486766180504,
        "mean_fdtd_silicon_absorption_proxy": 0.6087103328224176,
        "mean_fdtd_trench_field_leakage": 0.1304583724250464,
    },
    "bdti": {
        "max_abs_error_R": 0.018285819966702088,
        "max_abs_error_T": 0.00881966457550798,
        "max_abs_error_A": 0.019033790017604746,
        "mean_fdtd_silicon_absorption_proxy": 0.6104874325359292,
        "mean_fdtd_trench_field_leakage": 0.09597948824750697,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DTI crosstalk report.")
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
    if isinstance(value, BenchmarkSnapshot):
        return json_safe(value.__dict__)
    return value


def pixel_config(mode: str, width_um: float, depth_um: float) -> dict[str, Any]:
    pixel = copy.deepcopy(derive_parameters("generic_bsi", pitch=1.0))
    dti = pixel["layers"]["silicon"].setdefault("dti", {})
    enabled = width_um > 0.0 and depth_um > 0.0
    dti["enabled"] = enabled
    dti["mode"] = mode
    dti["width"] = float(width_um)
    dti["depth"] = float(depth_um)
    dti["material"] = "sio2"
    return pixel


def silicon_eps() -> complex:
    return complex(MaterialDB().get_epsilon("silicon", WAVELENGTH_UM))


def dti_mask_from_eps(eps_grid: np.ndarray, eps_si: complex) -> np.ndarray:
    return np.abs(eps_grid - eps_si) > 1e-6 * max(abs(eps_si), 1.0)


def geometry_metrics(mode: str, width_um: float, depth_um: float, nx: int = 400, ny: int = 400) -> dict[str, Any]:
    pixel = pixel_config(mode, width_um, depth_um)
    stack = PixelStack({"pixel": pixel})
    si_thickness = float(pixel["layers"]["silicon"]["thickness"])
    eps_si = silicon_eps()
    slices = [
        item
        for item in stack.get_layer_slices(WAVELENGTH_UM, nx=nx, ny=ny, n_lens_slices=4)
        if item.name.startswith("silicon")
    ]
    weighted_fraction = 0.0
    active_depth = 0.0
    max_area_fraction = 0.0
    slice_rows: list[dict[str, float | str]] = []
    for item in slices:
        mask = dti_mask_from_eps(item.eps_grid, eps_si)
        area_fraction = float(np.mean(mask))
        weighted_fraction += area_fraction * float(item.thickness)
        if area_fraction > 1e-6:
            active_depth += float(item.thickness)
        max_area_fraction = max(max_area_fraction, area_fraction)
        slice_rows.append(
            {
                "name": item.name,
                "z_start_um": float(item.z_start),
                "z_end_um": float(item.z_end),
                "thickness_um": float(item.thickness),
                "dti_area_fraction": area_fraction,
            }
        )
    effective_volume_fraction = weighted_fraction / max(si_thickness, 1e-12)
    return {
        "mode": mode,
        "width_um": float(width_um),
        "depth_um": float(depth_um),
        "silicon_thickness_um": si_thickness,
        "active_dti_depth_um": active_depth,
        "max_xy_dti_area_fraction": max_area_fraction,
        "effective_dti_volume_fraction": effective_volume_fraction,
        "open_silicon_volume_fraction": 1.0 - effective_volume_fraction,
        "slices": slice_rows,
    }


def width_sweep() -> list[dict[str, Any]]:
    return [
        geometry_metrics("fdti", width, depth_um=3.0)
        for width in WIDTH_VALUES
    ]


def depth_sweep() -> list[dict[str, Any]]:
    return [
        geometry_metrics("bdti", width_um=0.10, depth_um=depth)
        for depth in DEPTH_VALUES
    ]


def dti_xy_mask(
    mode: str,
    width_um: float,
    depth_um: float,
    nx: int = 220,
    ny: int = 220,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixel = pixel_config(mode, width_um, depth_um)
    stack = PixelStack({"pixel": pixel})
    si_cfg = pixel["layers"]["silicon"]
    eps_si = silicon_eps()
    x = np.linspace(0.0, stack.domain_size[0], nx)
    y = np.linspace(0.0, stack.domain_size[1], ny)
    eps_grid = stack._build_si_grid_at_depth(WAVELENGTH_UM, nx, ny, si_cfg, depth_from_top=0.05)
    mask = dti_mask_from_eps(eps_grid, eps_si).astype(float)
    return x, y, mask


def dti_xz_mask(
    mode: str,
    width_um: float,
    depth_um: float,
    nx: int = 220,
    ny: int = 220,
    nz: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixel = pixel_config(mode, width_um, depth_um)
    stack = PixelStack({"pixel": pixel})
    si_cfg = pixel["layers"]["silicon"]
    si_thickness = float(si_cfg["thickness"])
    eps_si = silicon_eps()
    x = np.linspace(0.0, stack.domain_size[0], nx)
    z = np.linspace(0.0, si_thickness, nz)
    mask = np.zeros((nz, nx), dtype=float)
    y = (np.arange(ny) + 0.5) * (stack.domain_size[1] / ny)
    y_index = int(np.argmin(np.abs(y - (0.5 * stack.pitch))))
    for zi, z_rel in enumerate(z):
        depth_from_top = si_thickness - float(z_rel)
        eps_grid = stack._build_si_grid_at_depth(WAVELENGTH_UM, nx, ny, si_cfg, depth_from_top)
        mask[zi, :] = dti_mask_from_eps(eps_grid[y_index], eps_si)
    return x, z, mask


def plot_geometry_sweeps(width_rows: list[dict[str, Any]], depth_rows: list[dict[str, Any]], outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    axes[0].plot(
        [row["width_um"] * 1000.0 for row in width_rows],
        [row["effective_dti_volume_fraction"] for row in width_rows],
        marker="o",
        linewidth=2.0,
        color="#2563eb",
    )
    axes[0].set_xlabel("FDTI width (nm)")
    axes[0].set_ylabel("effective DTI volume fraction")
    axes[0].set_title("Full-depth DTI width sweep")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        [row["depth_um"] for row in depth_rows],
        [row["effective_dti_volume_fraction"] for row in depth_rows],
        marker="s",
        linewidth=2.0,
        color="#16a34a",
        label="volume fraction",
    )
    axes[1].plot(
        [row["depth_um"] for row in depth_rows],
        [row["active_dti_depth_um"] / row["silicon_thickness_um"] for row in depth_rows],
        marker="^",
        linewidth=1.8,
        color="#f97316",
        label="active depth / silicon thickness",
    )
    axes[1].set_xlabel("BDTI depth from backside/top (um)")
    axes[1].set_ylabel("fraction")
    axes[1].set_title("BDTI depth sweep, width 100 nm")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("PixelStack DTI geometry sweep, generic 1.0 um BSI", fontsize=14)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_xz_masks(outpath: Path) -> None:
    cases = [
        ("none", "none", 0.0, 0.0),
        ("bdti_0p6", "bdti", 0.10, 0.6),
        ("fdti", "fdti", 0.10, 3.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.0), constrained_layout=True)
    for col, (label, mode, width, depth) in enumerate(cases):
        x, y, xy_mask = dti_xy_mask(mode, width, depth)
        axes[0, col].imshow(
            xy_mask,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            aspect="equal",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, col].set_title(label.replace("_", " "))
        axes[0, col].set_xlabel("x (um)")
        axes[0, col].grid(False)

        x, z, xz_mask = dti_xz_mask(mode, width, depth)
        axes[1, col].imshow(
            xz_mask,
            extent=[x.min(), x.max(), z.min(), z.max()],
            origin="lower",
            aspect="auto",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, col].set_xlabel("x (um)")
        axes[1, col].grid(False)
    axes[0, 0].set_ylabel("y (um), XY at 50 nm below Si top")
    axes[1, 0].set_ylabel("silicon z from bottom (um), XZ at y=0.5 um")
    fig.suptitle("Silicon DTI occupancy masks generated from PixelStack", fontsize=14)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_crosstalk_matrices(outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for ax, snapshot in zip(axes, PIXEL_BENCHMARK):
        mat = np.array(snapshot.matrix, dtype=float)
        im = ax.imshow(mat, cmap="magma", vmin=0.0, vmax=0.85)
        ax.set_xticks(range(len(PIXEL_LABELS)), PIXEL_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(len(PIXEL_LABELS)), PIXEL_LABELS)
        ax.set_title(snapshot.mode)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        ax.set_xlabel("collected pixel")
        ax.set_ylabel("source pixel")
    fig.colorbar(im, ax=axes, shrink=0.82, label="collection fraction")
    fig.suptitle("Representative 44x44x118, 3500-step scalar FDTD crosstalk snapshot", fontsize=14)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_benchmark_summary(outpath: Path) -> None:
    labels = [item.mode for item in PIXEL_BENCHMARK]
    x = np.arange(len(labels))
    width = 0.34
    self_values = [item.mean_self_collection_fraction for item in PIXEL_BENCHMARK]
    xt_values = [item.max_neighbor_crosstalk_fraction for item in PIXEL_BENCHMARK]

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, self_values, width, color="#2563eb", alpha=0.86, label="mean self collection")
    ax.bar(x + width / 2, xt_values, width, color="#dc2626", alpha=0.86, label="max neighbor crosstalk")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 0.7)
    ax.set_ylabel("fraction")
    ax.set_title("Representative scalar FDTD DTI crosstalk summary")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def width_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        ["width nm", "max XY DTI area", "effective DTI volume", "open Si volume"],
        [
            [
                row["width_um"] * 1000.0,
                row["max_xy_dti_area_fraction"],
                row["effective_dti_volume_fraction"],
                row["open_silicon_volume_fraction"],
            ]
            for row in rows
        ],
    )


def depth_table(rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        ["BDTI depth um", "active depth um", "effective DTI volume", "open Si volume"],
        [
            [
                row["depth_um"],
                row["active_dti_depth_um"],
                row["effective_dti_volume_fraction"],
                row["open_silicon_volume_fraction"],
            ]
            for row in rows
        ],
    )


def snapshot_table() -> str:
    return markdown_table(
        [
            "mode",
            "mean self collection",
            "max neighbor crosstalk",
            "mean PD signal",
            "energy tail change",
        ],
        [
            [
                item.mode,
                item.mean_self_collection_fraction,
                item.max_neighbor_crosstalk_fraction,
                item.mean_total_pd_signal,
                item.max_energy_tail_relative_change,
            ]
            for item in PIXEL_BENCHMARK
        ],
    )


def periodic_trench_table() -> str:
    return markdown_table(
        [
            "mode",
            "max abs dR",
            "max abs dT",
            "max abs dA",
            "Si absorption proxy",
            "trench field leakage",
        ],
        [
            [
                mode,
                item["max_abs_error_R"],
                item["max_abs_error_T"],
                item["max_abs_error_A"],
                item["mean_fdtd_silicon_absorption_proxy"],
                item["mean_fdtd_trench_field_leakage"],
            ]
            for mode, item in PERIODIC_TRENCH_SUMMARY.items()
        ],
    )


def write_reports(
    docs_root: Path,
    generated_on: str,
    width_rows: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
) -> None:
    reports = docs_root / "reports"
    reports_ko = docs_root / "ko" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    reports_ko.mkdir(parents=True, exist_ok=True)

    fdti_100 = next(row for row in width_rows if abs(row["width_um"] - 0.10) < 1e-9)
    bdti_060 = next(row for row in depth_rows if abs(row["depth_um"] - 0.60) < 1e-9)
    fdti_snapshot = next(item for item in PIXEL_BENCHMARK if item.mode == "fdti")
    bdti_snapshot = next(item for item in PIXEL_BENCHMARK if item.mode == "bdti_0p6um")
    xt_gap = bdti_snapshot.max_neighbor_crosstalk_fraction - fdti_snapshot.max_neighbor_crosstalk_fraction

    en = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# DTI Crosstalk Benchmark",
            "",
            f"_Generated on {generated_on} from PixelStack geometry sweeps and a representative scalar FDTD snapshot._",
            "",
            "This report separates cheap, reproducible DTI geometry evidence from the "
            "more expensive localized-source crosstalk benchmark. It is intended as a "
            "design-space gate before running longer vector FDTD or high-order RCWA jobs.",
            "",
            "## Executive summary",
            "",
            f"- A 100 nm full-depth DTI occupies {fmt(fdti_100['effective_dti_volume_fraction'])} "
            "of the silicon volume in the generic 1.0 um 2x2 BSI PixelStack geometry.",
            f"- A 100 nm, 0.6 um BDTI occupies {fmt(bdti_060['effective_dti_volume_fraction'])} "
            "of the silicon volume because only the backside/top portion is trenched.",
            f"- The representative 44x44x118, 3500-step scalar FDTD snapshot reports max "
            f"neighbor crosstalk {fmt(fdti_snapshot.max_neighbor_crosstalk_fraction)} for FDTI "
            f"and {fmt(bdti_snapshot.max_neighbor_crosstalk_fraction)} for BDTI, a gap of "
            f"{fmt(xt_gap)} absolute.",
            "- The periodic trench RCWA/FDTD alignment snapshot remains within about 3 "
            "percentage points in R/T/A for both FDTI and BDTI.",
            "",
            "::: warning Scope",
            "The crosstalk matrix is a scalar FDTD visual benchmark, not a production "
            "full-vector FDTD solve. Use it to compare geometry and normalization paths; "
            "use longer vector runs for final isolation claims.",
            ":::",
            "",
            "## Geometry Sweep",
            "",
            "![DTI geometry sweeps](/reports/dti-crosstalk/01_dti_geometry_sweeps.png)",
            "",
            "### FDTI width sweep",
            "",
            width_table(width_rows),
            "",
            "### BDTI depth sweep",
            "",
            depth_table(depth_rows),
            "",
            "## Silicon DTI Masks",
            "",
            "![DTI XY and XZ masks](/reports/dti-crosstalk/02_dti_xz_masks.png)",
            "",
            "## Representative Crosstalk Snapshot",
            "",
            "![DTI crosstalk matrices](/reports/dti-crosstalk/03_dti_crosstalk_matrices.png)",
            "",
            "![DTI crosstalk summary](/reports/dti-crosstalk/04_dti_crosstalk_summary.png)",
            "",
            snapshot_table(),
            "",
            "## Periodic Trench Alignment Snapshot",
            "",
            periodic_trench_table(),
            "",
            "## Interpretation",
            "",
            "- FDTI and BDTI have similar crosstalk in the representative coarse scalar "
            "snapshot because the run is primarily a path and normalization check.",
            "- The geometry sweep still shows the intended monotonic controls: wider FDTI "
            "increases silicon trench volume, and deeper BDTI approaches FDTI.",
            "- A production DTI report should extend this with wavelength-resolved, "
            "localized-source vector FDTD and a true width/depth/material crosstalk sweep.",
            "",
            "## Regeneration",
            "",
            "```powershell",
            "uv run python scripts\\generate_dti_crosstalk_report.py",
            "```",
            "",
            "Generated metrics are stored at "
            "`docs/public/reports/dti-crosstalk/dti_crosstalk_metrics.json`.",
            "",
        ]
    )

    ko = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# DTI Crosstalk Benchmark 리포트",
            "",
            f"_생성일: {generated_on}. PixelStack geometry sweep와 representative scalar FDTD snapshot에서 생성._",
            "",
            "이 리포트는 빠르게 재생성 가능한 DTI geometry evidence와 비용이 큰 localized-source "
            "crosstalk benchmark를 분리한다. 긴 vector FDTD 또는 high-order RCWA를 돌리기 전 "
            "design-space gate로 사용하기 위한 문서다.",
            "",
            "## 요약",
            "",
            f"- 100 nm full-depth DTI는 generic 1.0 um 2x2 BSI PixelStack geometry에서 "
            f"silicon volume의 {fmt(fdti_100['effective_dti_volume_fraction'])}를 차지한다.",
            f"- 100 nm, 0.6 um BDTI는 backside/top 일부만 trench 처리되므로 silicon volume의 "
            f"{fmt(bdti_060['effective_dti_volume_fraction'])}를 차지한다.",
            f"- representative 44x44x118, 3500-step scalar FDTD snapshot의 max neighbor "
            f"crosstalk는 FDTI {fmt(fdti_snapshot.max_neighbor_crosstalk_fraction)}, "
            f"BDTI {fmt(bdti_snapshot.max_neighbor_crosstalk_fraction)}이며 gap은 "
            f"{fmt(xt_gap)} absolute다.",
            "- periodic trench RCWA/FDTD alignment snapshot은 FDTI와 BDTI 모두에서 R/T/A가 "
            "대략 3 percentage point 이내로 맞는다.",
            "",
            "::: warning 범위",
            "crosstalk matrix는 scalar FDTD visual benchmark이지 production full-vector FDTD "
            "solve가 아니다. geometry와 normalization path 비교용으로 사용하고, 최종 isolation "
            "claim에는 더 긴 vector run이 필요하다.",
            ":::",
            "",
            "## Geometry sweep",
            "",
            "![DTI geometry sweeps](/reports/dti-crosstalk/01_dti_geometry_sweeps.png)",
            "",
            "### FDTI width sweep",
            "",
            width_table(width_rows),
            "",
            "### BDTI depth sweep",
            "",
            depth_table(depth_rows),
            "",
            "## Silicon DTI mask",
            "",
            "![DTI XY and XZ masks](/reports/dti-crosstalk/02_dti_xz_masks.png)",
            "",
            "## Representative crosstalk snapshot",
            "",
            "![DTI crosstalk matrices](/reports/dti-crosstalk/03_dti_crosstalk_matrices.png)",
            "",
            "![DTI crosstalk summary](/reports/dti-crosstalk/04_dti_crosstalk_summary.png)",
            "",
            snapshot_table(),
            "",
            "## Periodic trench alignment snapshot",
            "",
            periodic_trench_table(),
            "",
            "## 해석",
            "",
            "- 대표 coarse scalar snapshot에서 FDTI와 BDTI의 crosstalk가 비슷한 이유는 이 run이 "
            "주로 path와 normalization check 역할을 하기 때문이다.",
            "- geometry sweep은 의도한 monotonic control을 보여준다. FDTI가 넓어질수록 silicon "
            "trench volume이 증가하고, BDTI가 깊어질수록 FDTI에 가까워진다.",
            "- production DTI report는 wavelength-resolved localized-source vector FDTD와 "
            "실제 width/depth/material crosstalk sweep으로 확장해야 한다.",
            "",
            "## 재생성",
            "",
            "```powershell",
            "uv run python scripts\\generate_dti_crosstalk_report.py",
            "```",
            "",
            "생성 metric은 "
            "`docs/public/reports/dti-crosstalk/dti_crosstalk_metrics.json`에 저장된다.",
            "",
        ]
    )

    (reports / "dti-crosstalk-benchmark.md").write_text(en, encoding="utf-8")
    (reports_ko / "dti-crosstalk-benchmark.md").write_text(ko, encoding="utf-8")


def main() -> None:
    args = parse_args()
    docs_root = args.docs.resolve()
    public_dir = docs_root / "public" / "reports" / "dti-crosstalk"
    public_dir.mkdir(parents=True, exist_ok=True)

    width_rows = width_sweep()
    depth_rows = depth_sweep()
    plot_geometry_sweeps(width_rows, depth_rows, public_dir / "01_dti_geometry_sweeps.png")
    plot_xz_masks(public_dir / "02_dti_xz_masks.png")
    plot_crosstalk_matrices(public_dir / "03_dti_crosstalk_matrices.png")
    plot_benchmark_summary(public_dir / "04_dti_crosstalk_summary.png")

    metrics = {
        "generated_on": args.date,
        "benchmark": {
            "pixel": "generic_bsi",
            "pitch_um": 1.0,
            "wavelength_um": WAVELENGTH_UM,
            "geometry_grid": [400, 400],
            "representative_scalar_fdtd": {
                "grid": [44, 44, 118],
                "steps": 3500,
                "source_set": "all",
                "note": "Snapshot copied as fixed evidence from the convergence ladder.",
            },
        },
        "width_sweep": width_rows,
        "depth_sweep": depth_rows,
        "pixel_benchmark_snapshot": PIXEL_BENCHMARK,
        "periodic_trench_summary": PERIODIC_TRENCH_SUMMARY,
    }
    (public_dir / "dti_crosstalk_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_reports(docs_root, args.date, width_rows, depth_rows)
    print(f"Wrote DTI crosstalk report and assets to {public_dir}")


if __name__ == "__main__":
    main()
