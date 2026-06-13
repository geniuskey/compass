#!/usr/bin/env python3
"""Generate the BARL optimization benchmark report.

The benchmark uses the COMPASS TMM adapter as a fast planar thin-film gate for
BARL design. It intentionally reports reflection-centric metrics because TMM
does not model lateral color-filter geometry, microlens focusing, or crosstalk.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
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

from compass.geometry.sample_pixels import derive_parameters  # noqa: E402
from compass.materials.database import MaterialDB  # noqa: E402
from compass.runners.single_run import SingleRunner  # noqa: E402

DOCS = ROOT / "docs"
PUBLIC_DIR = DOCS / "public" / "reports" / "barl-optimization"
WAVELENGTHS = np.round(np.arange(0.40, 0.7001, 0.01), 4)


@dataclass(frozen=True)
class Design:
    key: str
    label: str
    layers: list[dict[str, Any]]
    role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BARL optimization report.")
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


def source_config() -> dict[str, Any]:
    return {
        "wavelength": {
            "mode": "list",
            "values": [float(wl) for wl in WAVELENGTHS],
        },
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "unpolarized",
    }


def solver_config() -> dict[str, Any]:
    return {
        "name": "tmm",
        "type": "tmm",
        "params": {
            "polarization_average": True,
        },
    }


def pixel_with_barl(layers: list[dict[str, Any]]) -> dict[str, Any]:
    pixel = copy.deepcopy(derive_parameters("generic_bsi", pitch=1.0))
    pixel["layers"]["barl"]["layers"] = copy.deepcopy(layers)
    return pixel


def run_tmm_barl(layers: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = {
        "pixel": pixel_with_barl(layers),
        "compute": {"backend": "cpu"},
        "solver": solver_config(),
        "source": source_config(),
    }
    started = time.perf_counter()
    result = SingleRunner.run(cfg)
    runtime_s = time.perf_counter() - started
    reflection = np.asarray(result.reflection, dtype=float)
    transmission = np.asarray(result.transmission, dtype=float)
    absorption = np.asarray(result.absorption, dtype=float)
    idx_550 = int(np.argmin(np.abs(WAVELENGTHS - 0.55)))
    return {
        "runtime_s": runtime_s,
        "wavelength_um": WAVELENGTHS.copy(),
        "reflection": reflection,
        "transmission": transmission,
        "absorption": absorption,
        "mean_reflection": float(np.mean(reflection)),
        "max_reflection": float(np.max(reflection)),
        "reflection_550": float(reflection[idx_550]),
        "mean_absorption": float(np.mean(absorption)),
        "absorption_550": float(absorption[idx_550]),
        "energy_residual": float(np.max(np.abs(reflection + transmission + absorption - 1.0))),
    }


def layer_summary(layers: list[dict[str, Any]]) -> str:
    if not layers:
        return "none"
    return " / ".join(
        f"{item['material']} {float(item['thickness']) * 1000:.0f} nm"
        for item in layers
    )


def total_thickness_nm(layers: list[dict[str, Any]]) -> float:
    return float(sum(float(item["thickness"]) for item in layers) * 1000.0)


def quarter_wave_info() -> dict[str, float]:
    db = MaterialDB()
    n_cf, _ = db.get_nk("cf_green", 0.55)
    n_si, _ = db.get_nk("silicon", 0.55)
    n_si3n4, _ = db.get_nk("si3n4", 0.55)
    n_hfo2, _ = db.get_nk("hfo2", 0.55)
    ideal_n = math.sqrt(float(n_cf) * float(n_si))
    return {
        "n_cf_green_550": float(n_cf),
        "n_silicon_550": float(n_si),
        "n_si3n4_550": float(n_si3n4),
        "n_hfo2_550": float(n_hfo2),
        "ideal_arc_n_550": ideal_n,
        "si3n4_quarter_wave_um": float(0.55 / (4.0 * n_si3n4)),
        "hfo2_quarter_wave_um": float(0.55 / (4.0 * n_hfo2)),
    }


def single_layer_sweep(material: str = "si3n4") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for thickness in np.arange(0.020, 0.1501, 0.0025):
        layers = [{"material": material, "thickness": float(thickness)}]
        result = run_tmm_barl(layers)
        rows.append(
            {
                "material": material,
                "thickness_um": float(thickness),
                **result,
            }
        )
    return rows


def two_layer_sweep() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sio2_values = np.arange(0.005, 0.0701, 0.005)
    hfo2_values = np.arange(0.005, 0.0701, 0.005)
    for sio2_t in sio2_values:
        for hfo2_t in hfo2_values:
            layers = [
                {"material": "sio2", "thickness": float(sio2_t)},
                {"material": "hfo2", "thickness": float(hfo2_t)},
            ]
            result = run_tmm_barl(layers)
            rows.append(
                {
                    "sio2_thickness_um": float(sio2_t),
                    "hfo2_thickness_um": float(hfo2_t),
                    **result,
                }
            )
    return rows


def best_by_mean_reflection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda item: item["mean_reflection"])


def candidate_designs(best_single: dict[str, Any], best_two: dict[str, Any]) -> list[Design]:
    baseline = derive_parameters("generic_bsi", pitch=1.0)["layers"]["barl"]["layers"]
    return [
        Design("none", "No BARL", [], "baseline"),
        Design("default", "Default 4-layer", copy.deepcopy(baseline), "sample default"),
        Design(
            "si3n4_qw",
            "Si3N4 quarter-wave",
            [{"material": "si3n4", "thickness": quarter_wave_info()["si3n4_quarter_wave_um"]}],
            "analytic",
        ),
        Design(
            "best_si3n4",
            "Best single Si3N4",
            [
                {
                    "material": "si3n4",
                    "thickness": best_single["thickness_um"],
                }
            ],
            "sweep best",
        ),
        Design(
            "best_sio2_hfo2",
            "Best SiO2/HfO2",
            [
                {"material": "sio2", "thickness": best_two["sio2_thickness_um"]},
                {"material": "hfo2", "thickness": best_two["hfo2_thickness_um"]},
            ],
            "sweep best",
        ),
    ]


def evaluate_designs(designs: list[Design]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for design in designs:
        result = run_tmm_barl(design.layers)
        rows.append(
            {
                "key": design.key,
                "label": design.label,
                "role": design.role,
                "layers": design.layers,
                "layer_summary": layer_summary(design.layers),
                "total_thickness_nm": total_thickness_nm(design.layers),
                **result,
            }
        )
    return rows


def plot_design_spectra(design_rows: list[dict[str, Any]], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    colors = {
        "none": "#475569",
        "default": "#dc2626",
        "si3n4_qw": "#f97316",
        "best_si3n4": "#16a34a",
        "best_sio2_hfo2": "#2563eb",
    }
    for row in design_rows:
        ax.plot(
            row["wavelength_um"] * 1000.0,
            row["reflection"],
            linewidth=2.0,
            color=colors.get(row["key"]),
            label=f"{row['label']} (mean R={fmt(row['mean_reflection'], 3)})",
        )
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.set_title("BARL reflectance spectra, TMM planar green-stack proxy")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_single_sweep(rows: list[dict[str, Any]], qw_um: float, outpath: Path) -> None:
    best = best_by_mean_reflection(rows)
    x_nm = np.array([row["thickness_um"] for row in rows]) * 1000.0
    mean_r = np.array([row["mean_reflection"] for row in rows])
    r_550 = np.array([row["reflection_550"] for row in rows])

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    ax.plot(x_nm, mean_r, color="#2563eb", linewidth=2.0, label="mean R, 400-700 nm")
    ax.plot(x_nm, r_550, color="#f97316", linewidth=1.8, label="R at 550 nm")
    ax.axvline(qw_um * 1000.0, color="#64748b", linestyle="--", label="quarter-wave")
    ax.axvline(best["thickness_um"] * 1000.0, color="#16a34a", linestyle=":", label="sweep optimum")
    ax.set_xlabel("Si3N4 thickness (nm)")
    ax.set_ylabel("reflectance")
    ax.set_title("Single-layer Si3N4 BARL thickness sweep")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_two_layer_heatmap(rows: list[dict[str, Any]], outpath: Path) -> None:
    sio2_values = sorted({row["sio2_thickness_um"] for row in rows})
    hfo2_values = sorted({row["hfo2_thickness_um"] for row in rows})
    grid = np.full((len(hfo2_values), len(sio2_values)), np.nan)
    for row in rows:
        i = hfo2_values.index(row["hfo2_thickness_um"])
        j = sio2_values.index(row["sio2_thickness_um"])
        grid[i, j] = row["mean_reflection"]
    best = best_by_mean_reflection(rows)

    fig, ax = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True)
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(sio2_values)), [f"{v * 1000:.0f}" for v in sio2_values])
    ax.set_yticks(range(len(hfo2_values)), [f"{v * 1000:.0f}" for v in hfo2_values])
    ax.set_xlabel("SiO2 thickness (nm)")
    ax.set_ylabel("HfO2 thickness (nm)")
    ax.set_title("Two-layer SiO2/HfO2 sweep: mean reflectance")
    best_x = sio2_values.index(best["sio2_thickness_um"])
    best_y = hfo2_values.index(best["hfo2_thickness_um"])
    ax.scatter([best_x], [best_y], marker="x", s=110, color="#dc2626", linewidths=2.2)
    fig.colorbar(im, ax=ax, label="mean R over 400-700 nm")
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_scorecard(design_rows: list[dict[str, Any]], outpath: Path) -> None:
    labels = [row["label"] for row in design_rows]
    x = np.arange(len(labels))
    width = 0.34
    mean_r = [row["mean_reflection"] for row in design_rows]
    max_r = [row["max_reflection"] for row in design_rows]

    fig, ax = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    ax.bar(x - width / 2, mean_r, width, label="mean R", color="#2563eb", alpha=0.86)
    ax.bar(x + width / 2, max_r, width, label="max R", color="#f97316", alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("reflectance")
    ax.set_title("BARL design scorecard")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def design_table(design_rows: list[dict[str, Any]]) -> str:
    return markdown_table(
        [
            "Design",
            "role",
            "layers",
            "total nm",
            "mean R",
            "max R",
            "R@550",
            "mean A",
            "energy residual",
        ],
        [
            [
                row["label"],
                row["role"],
                row["layer_summary"],
                row["total_thickness_nm"],
                row["mean_reflection"],
                row["max_reflection"],
                row["reflection_550"],
                row["mean_absorption"],
                row["energy_residual"],
            ]
            for row in design_rows
        ],
    )


def write_reports(
    docs_root: Path,
    generated_on: str,
    qw: dict[str, float],
    single_rows: list[dict[str, Any]],
    two_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
) -> None:
    reports = docs_root / "reports"
    reports_ko = docs_root / "ko" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    reports_ko.mkdir(parents=True, exist_ok=True)

    best_single = best_by_mean_reflection(single_rows)
    best_two = best_by_mean_reflection(two_rows)
    no_barl = next(row for row in design_rows if row["key"] == "none")
    best_design = min(design_rows, key=lambda row: row["mean_reflection"])
    mean_r_delta = no_barl["mean_reflection"] - best_design["mean_reflection"]

    en = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# BARL Optimization Benchmark",
            "",
            f"_Generated on {generated_on} with the COMPASS TMM solver on the generic 1.0 um BSI stack._",
            "",
            "This report turns the BARL cookbook recipe into a reproducible benchmark. "
            "It sweeps single-layer Si3N4 thickness, a two-layer SiO2/HfO2 grid, and "
            "then compares representative designs on the same 400-700 nm wavelength grid.",
            "",
            "## Executive summary",
            "",
            f"- The 550 nm ideal ARC index between `cf_green` and silicon is "
            f"{fmt(qw['ideal_arc_n_550'])}; Si3N4 gives a quarter-wave thickness of "
            f"{fmt(qw['si3n4_quarter_wave_um'] * 1000.0)} nm.",
            f"- The best single-layer Si3N4 sweep point is "
            f"{fmt(best_single['thickness_um'] * 1000.0)} nm with mean R="
            f"{fmt(best_single['mean_reflection'])}.",
            f"- The best two-layer SiO2/HfO2 grid point is SiO2 "
            f"{fmt(best_two['sio2_thickness_um'] * 1000.0)} nm / HfO2 "
            f"{fmt(best_two['hfo2_thickness_um'] * 1000.0)} nm with mean R="
            f"{fmt(best_two['mean_reflection'])}.",
            f"- Best design in this candidate set reduces mean reflection by "
            f"{fmt(mean_r_delta)} absolute versus no BARL.",
            "",
            "::: warning Planar proxy",
            "This is a TMM planar green-stack benchmark. It is appropriate for BARL "
            "screening and reflection trends, but it does not include lateral Bayer "
            "geometry, microlens focusing, metal-grid diffraction, or crosstalk.",
            ":::",
            "",
            "## Candidate Spectra",
            "",
            "![BARL design spectra](/reports/barl-optimization/01_barl_design_spectra.png)",
            "",
            "## Single-layer Sweep",
            "",
            "![Single-layer BARL sweep](/reports/barl-optimization/02_single_layer_sweep.png)",
            "",
            "## Two-layer Sweep",
            "",
            "![Two-layer BARL heatmap](/reports/barl-optimization/03_hfo2_sio2_heatmap.png)",
            "",
            "## Design Scorecard",
            "",
            "![BARL scorecard](/reports/barl-optimization/04_barl_design_scorecard.png)",
            "",
            design_table(design_rows),
            "",
            "## Interpretation",
            "",
            "- BARL tuning should optimize broadband reflection, not only R at 550 nm. "
            "A quarter-wave layer is a useful seed, but it is not automatically the "
            "broadband optimum once color-filter and planarization phases are included.",
            "- The default sample BARL is an illustrative process stack, not a guaranteed "
            "optimum. This report makes that explicit by comparing it against simple "
            "swept alternatives.",
            "- The two-layer optimum sits on the lower SiO2 sweep boundary, so the next "
            "local search should extend toward thinner SiO2 or test HfO2-only variants.",
            "- After a TMM BARL candidate is selected, run a patterned RCWA check because "
            "the metal grid and microlens can move the apparent optimum.",
            "",
            "## Regeneration",
            "",
            "```powershell",
            "uv run python scripts\\generate_barl_optimization_report.py",
            "```",
            "",
            "Generated metrics are stored at "
            "`docs/public/reports/barl-optimization/barl_optimization_metrics.json`.",
            "",
        ]
    )

    ko = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# BARL Optimization Benchmark 리포트",
            "",
            f"_생성일: {generated_on}. generic 1.0 um BSI stack에서 COMPASS TMM solver로 생성._",
            "",
            "이 리포트는 BARL cookbook recipe를 재현 가능한 benchmark로 승격한다. "
            "single-layer Si3N4 두께, two-layer SiO2/HfO2 grid를 sweep하고, "
            "대표 design을 같은 400-700 nm wavelength grid에서 비교한다.",
            "",
            "## 요약",
            "",
            f"- `cf_green`과 silicon 사이의 550 nm ideal ARC index는 "
            f"{fmt(qw['ideal_arc_n_550'])}이고, Si3N4 quarter-wave thickness는 "
            f"{fmt(qw['si3n4_quarter_wave_um'] * 1000.0)} nm다.",
            f"- single-layer Si3N4 sweep 최적점은 "
            f"{fmt(best_single['thickness_um'] * 1000.0)} nm이며 mean R="
            f"{fmt(best_single['mean_reflection'])}다.",
            f"- two-layer SiO2/HfO2 grid 최적점은 SiO2 "
            f"{fmt(best_two['sio2_thickness_um'] * 1000.0)} nm / HfO2 "
            f"{fmt(best_two['hfo2_thickness_um'] * 1000.0)} nm이며 mean R="
            f"{fmt(best_two['mean_reflection'])}다.",
            f"- 이 candidate set의 최선 design은 no BARL 대비 mean reflection을 "
            f"{fmt(mean_r_delta)} absolute 줄인다.",
            "",
            "::: warning Planar proxy",
            "이 리포트는 TMM planar green-stack benchmark다. BARL screening과 reflection "
            "trend에는 적합하지만 lateral Bayer geometry, microlens focusing, metal-grid "
            "diffraction, crosstalk는 포함하지 않는다.",
            ":::",
            "",
            "## Candidate spectra",
            "",
            "![BARL design spectra](/reports/barl-optimization/01_barl_design_spectra.png)",
            "",
            "## Single-layer sweep",
            "",
            "![Single-layer BARL sweep](/reports/barl-optimization/02_single_layer_sweep.png)",
            "",
            "## Two-layer sweep",
            "",
            "![Two-layer BARL heatmap](/reports/barl-optimization/03_hfo2_sio2_heatmap.png)",
            "",
            "## Design scorecard",
            "",
            "![BARL scorecard](/reports/barl-optimization/04_barl_design_scorecard.png)",
            "",
            design_table(design_rows),
            "",
            "## 해석",
            "",
            "- BARL tuning은 R@550 하나가 아니라 broadband reflection을 최적화해야 한다. "
            "Quarter-wave layer는 좋은 seed지만 color-filter와 planarization phase가 포함되면 "
            "자동으로 broadband optimum이 되지는 않는다.",
            "- 기본 sample BARL은 예시 process stack이지 보장된 optimum이 아니다. 이 리포트는 "
            "간단한 sweep alternative와 비교해 그 점을 명시한다.",
            "- two-layer optimum은 SiO2 sweep의 lower boundary에 걸려 있다. 다음 local "
            "search에서는 더 얇은 SiO2 또는 HfO2-only variant를 확인하는 것이 좋다.",
            "- TMM BARL candidate를 고른 뒤에는 patterned RCWA check를 돌려야 한다. metal grid와 "
            "microlens가 apparent optimum을 이동시킬 수 있기 때문이다.",
            "",
            "## 재생성",
            "",
            "```powershell",
            "uv run python scripts\\generate_barl_optimization_report.py",
            "```",
            "",
            "생성 metric은 "
            "`docs/public/reports/barl-optimization/barl_optimization_metrics.json`에 저장된다.",
            "",
        ]
    )

    (reports / "barl-optimization-benchmark.md").write_text(en, encoding="utf-8")
    (reports_ko / "barl-optimization-benchmark.md").write_text(ko, encoding="utf-8")


def main() -> None:
    args = parse_args()
    docs_root = args.docs.resolve()
    public_dir = docs_root / "public" / "reports" / "barl-optimization"
    public_dir.mkdir(parents=True, exist_ok=True)

    qw = quarter_wave_info()
    single_rows = single_layer_sweep()
    best_single = best_by_mean_reflection(single_rows)
    two_rows = two_layer_sweep()
    best_two = best_by_mean_reflection(two_rows)
    designs = candidate_designs(best_single, best_two)
    design_rows = evaluate_designs(designs)

    plot_design_spectra(design_rows, public_dir / "01_barl_design_spectra.png")
    plot_single_sweep(single_rows, qw["si3n4_quarter_wave_um"], public_dir / "02_single_layer_sweep.png")
    plot_two_layer_heatmap(two_rows, public_dir / "03_hfo2_sio2_heatmap.png")
    plot_scorecard(design_rows, public_dir / "04_barl_design_scorecard.png")

    metrics = {
        "generated_on": args.date,
        "benchmark": {
            "pixel": "generic_bsi",
            "pitch_um": 1.0,
            "solver": "tmm",
            "wavelength_um": WAVELENGTHS,
            "objective": "minimize mean reflectance over 400-700 nm",
        },
        "quarter_wave": qw,
        "single_layer_sweep": single_rows,
        "two_layer_sweep": two_rows,
        "candidate_designs": design_rows,
    }
    (public_dir / "barl_optimization_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_reports(docs_root, args.date, qw, single_rows, two_rows, design_rows)
    print(f"Wrote BARL optimization report and assets to {public_dir}")


if __name__ == "__main__":
    main()
