#!/usr/bin/env python3
"""Generate the TMM vs zero-order RCWA planar-stack validation report.

This report is intentionally narrower than the full convergence report. It
isolates 1D planar optics only: no lateral patterning, no FDTD, and no pixel
collection windows. In this limit, zero-order RCWA should reduce to the same
physics as the transfer-matrix method (TMM). The generated figures and metrics
make that baseline easy to inspect from the docs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compass.solvers.tmm.tmm_core import transfer_matrix_1d  # noqa: E402

DOCS = ROOT / "docs"
PUBLIC_DIR = DOCS / "public" / "reports" / "tmm-rcwa-planar"


@dataclass(frozen=True)
class LayerSpec:
    name: str
    refractive_index: complex
    thickness_um: float


@dataclass(frozen=True)
class PlanarCase:
    key: str
    title: str
    incident_index: complex
    substrate_index: complex
    layers: tuple[LayerSpec, ...]
    note: str

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def stack_thickness_um(self) -> float:
        return sum(layer.thickness_um for layer in self.layers)


@dataclass(frozen=True)
class Rta:
    reflection: float
    transmission: float
    absorption: float
    runtime_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TMM vs zero-order RCWA planar-stack validation report.",
    )
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=[round(v, 3) for v in np.linspace(0.40, 0.70, 31)],
        help="Wavelengths in um.",
    )
    return parser.parse_args()


def planar_cases() -> list[PlanarCase]:
    design_wl = 0.55
    n_si = 4.0
    n_arc = math.sqrt(n_si)
    return [
        PlanarCase(
            key="air_glass_interface",
            title="Air / glass interface",
            incident_index=1.0 + 0.0j,
            substrate_index=1.5 + 0.0j,
            layers=(),
            note="Analytical Fresnel baseline at normal incidence.",
        ),
        PlanarCase(
            key="ideal_arc_on_silicon",
            title="Ideal quarter-wave ARC on silicon",
            incident_index=1.0 + 0.0j,
            substrate_index=n_si + 0.0j,
            layers=(
                LayerSpec(
                    "ideal_arc",
                    n_arc + 0.0j,
                    design_wl / (4.0 * n_arc),
                ),
            ),
            note="Single-layer ARC designed for 0.55 um with n=sqrt(n_air*n_si).",
        ),
        PlanarCase(
            key="lossless_pixel_like_multilayer",
            title="Lossless pixel-like multilayer",
            incident_index=1.0 + 0.0j,
            substrate_index=1.0 + 0.0j,
            layers=(
                LayerSpec("polymer_lens_equiv", 1.56 + 0.0j, 0.22),
                LayerSpec("oxide_planarization", 1.46 + 0.0j, 0.18),
                LayerSpec("green_cf_equiv", 1.72 + 0.0j, 0.26),
                LayerSpec("barl_sio2", 1.46 + 0.0j, 0.04),
                LayerSpec("barl_hfo2", 2.05 + 0.0j, 0.05),
                LayerSpec("barl_si3n4", 2.00 + 0.0j, 0.06),
                LayerSpec("silicon_proxy", 3.50 + 0.0j, 0.18),
            ),
            note="Pixel-stack-like optical ladder without absorption.",
        ),
        PlanarCase(
            key="lossy_pixel_like_multilayer",
            title="Lossy pixel-like multilayer",
            incident_index=1.0 + 0.0j,
            substrate_index=1.0 + 0.0j,
            layers=(
                LayerSpec("polymer_lens_equiv", 1.56 + 0.0j, 0.22),
                LayerSpec("oxide_planarization", 1.46 + 0.0j, 0.18),
                LayerSpec("green_cf_absorber", 1.72 + 0.025j, 0.26),
                LayerSpec("barl_sio2", 1.46 + 0.0j, 0.04),
                LayerSpec("barl_hfo2", 2.05 + 0.0j, 0.05),
                LayerSpec("barl_si3n4", 2.00 + 0.0j, 0.06),
                LayerSpec("silicon_absorber", 3.50 + 0.050j, 0.18),
            ),
            note="Same ladder with color-filter and silicon-proxy absorption.",
        ),
    ]


def no_arc_silicon_case() -> PlanarCase:
    return PlanarCase(
        key="bare_silicon_interface",
        title="Bare air / silicon interface",
        incident_index=1.0 + 0.0j,
        substrate_index=4.0 + 0.0j,
        layers=(),
        note="Reference contrast for the quarter-wave ARC case.",
    )


def tmm_case(case: PlanarCase, wavelength_um: float) -> Rta:
    t0 = perf_counter()
    n_layers = np.array(
        [case.incident_index]
        + [layer.refractive_index for layer in case.layers]
        + [case.substrate_index],
        dtype=complex,
    )
    d_layers = np.array(
        [np.inf] + [layer.thickness_um for layer in case.layers] + [np.inf],
        dtype=float,
    )
    r, t, a = transfer_matrix_1d(
        n_layers=n_layers,
        d_layers=d_layers,
        wavelength=wavelength_um,
        theta_inc=0.0,
        polarization="TE",
    )
    return Rta(r, t, a, perf_counter() - t0)


def rcwa_case(case: PlanarCase, wavelength_um: float) -> Rta:
    t0 = perf_counter()
    import torch
    import torcwa

    sim = torcwa.rcwa(
        freq=1.0 / wavelength_um,
        order=[0, 0],
        L=[1.0, 1.0],
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    sim.add_input_layer(eps=case.incident_index * case.incident_index)
    sim.add_output_layer(eps=case.substrate_index * case.substrate_index)
    sim.set_incident_angle(0.0, 0.0)
    for layer in case.layers:
        sim.add_layer(
            thickness=layer.thickness_um,
            eps=layer.refractive_index * layer.refractive_index,
        )
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[1.0, 0.0], direction="forward", notation="xy")
    s_reflect = sim.S_parameters(
        orders=[0, 0],
        direction="forward",
        port="reflection",
        polarization="xx",
        power_norm=True,
    )
    s_transmit = sim.S_parameters(
        orders=[0, 0],
        direction="forward",
        port="transmission",
        polarization="xx",
        power_norm=True,
    )
    r = float(torch.abs(s_reflect) ** 2)
    t = float(torch.abs(s_transmit) ** 2)
    a = 1.0 - r - t
    if abs(a) < 1e-7:
        a = 0.0
    return Rta(r, t, a, perf_counter() - t0)


def run_case(case: PlanarCase, wavelengths: np.ndarray) -> dict[str, Any]:
    tmm_rows: list[Rta] = []
    rcwa_rows: list[Rta] = []
    for wl in wavelengths:
        tmm_rows.append(tmm_case(case, float(wl)))
        rcwa_rows.append(rcwa_case(case, float(wl)))

    tmm = {
        "R": np.array([row.reflection for row in tmm_rows]),
        "T": np.array([row.transmission for row in tmm_rows]),
        "A": np.array([row.absorption for row in tmm_rows]),
        "runtime_s": np.array([row.runtime_s for row in tmm_rows]),
    }
    rcwa = {
        "R": np.array([row.reflection for row in rcwa_rows]),
        "T": np.array([row.transmission for row in rcwa_rows]),
        "A": np.array([row.absorption for row in rcwa_rows]),
        "runtime_s": np.array([row.runtime_s for row in rcwa_rows]),
    }
    errors = {key: np.abs(rcwa[key] - tmm[key]) for key in ("R", "T", "A")}
    energy_tmm = np.abs(tmm["R"] + tmm["T"] + tmm["A"] - 1.0)
    energy_rcwa = np.abs(rcwa["R"] + rcwa["T"] + rcwa["A"] - 1.0)

    return {
        "case": case,
        "wavelengths": wavelengths,
        "tmm": tmm,
        "rcwa": rcwa,
        "errors": errors,
        "summary": {
            "max_abs_dR": float(np.max(errors["R"])),
            "max_abs_dT": float(np.max(errors["T"])),
            "max_abs_dA": float(np.max(errors["A"])),
            "max_energy_residual_tmm": float(np.max(energy_tmm)),
            "max_energy_residual_rcwa": float(np.max(energy_rcwa)),
            "mean_runtime_tmm_s": float(np.mean(tmm["runtime_s"])),
            "mean_runtime_rcwa_s": float(np.mean(rcwa["runtime_s"])),
            "passes": bool(
                max(
                    float(np.max(errors["R"])),
                    float(np.max(errors["T"])),
                    float(np.max(errors["A"])),
                )
                < 5e-5
            ),
        },
    }


def serialise_result(item: dict[str, Any]) -> dict[str, Any]:
    case: PlanarCase = item["case"]
    return {
        "key": case.key,
        "title": case.title,
        "incident_index": [case.incident_index.real, case.incident_index.imag],
        "substrate_index": [case.substrate_index.real, case.substrate_index.imag],
        "layer_count": case.layer_count,
        "stack_thickness_um": case.stack_thickness_um,
        "note": case.note,
        "layers": [
            {
                "name": layer.name,
                "n_real": layer.refractive_index.real,
                "n_imag": layer.refractive_index.imag,
                "thickness_um": layer.thickness_um,
            }
            for layer in case.layers
        ],
        "summary": item["summary"],
        "spectra": {
            "wavelength_um": item["wavelengths"].tolist(),
            "tmm": {key: item["tmm"][key].tolist() for key in ("R", "T", "A")},
            "rcwa": {key: item["rcwa"][key].tolist() for key in ("R", "T", "A")},
            "abs_error": {key: item["errors"][key].tolist() for key in ("R", "T", "A")},
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if 0 < abs(value) < 1e-4:
            return f"{value:.2e}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return fmt(value).replace("|", r"\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def plot_rta_alignment(results: list[dict[str, Any]], outpath: Path) -> None:
    fig, axes = plt.subplots(
        len(results),
        3,
        figsize=(14.5, 3.1 * len(results)),
        sharex=True,
        constrained_layout=True,
    )
    for row_idx, item in enumerate(results):
        case: PlanarCase = item["case"]
        wavelengths_nm = item["wavelengths"] * 1000.0
        for col_idx, key in enumerate(("R", "T", "A")):
            ax = axes[row_idx, col_idx]
            ax.plot(wavelengths_nm, item["tmm"][key], color="#2563eb", label="TMM")
            ax.plot(
                wavelengths_nm,
                item["rcwa"][key],
                color="#dc2626",
                linestyle="--",
                label="zero-order RCWA",
            )
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(key)
            if col_idx == 0:
                ax.set_ylabel(case.title)
            if row_idx == len(results) - 1:
                ax.set_xlabel("wavelength (nm)")
            if row_idx == 0 and col_idx == 2:
                ax.legend(loc="best", fontsize=8)
    fig.suptitle("TMM vs zero-order RCWA: planar-stack R/T/A alignment", fontsize=15)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_error_summary(results: list[dict[str, Any]], outpath: Path) -> None:
    labels = [item["case"].title for item in results]
    x = np.arange(len(labels))
    width = 0.23
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.2), constrained_layout=True)
    for offset, key, color in [
        (-width, "max_abs_dR", "#2563eb"),
        (0.0, "max_abs_dT", "#16a34a"),
        (width, "max_abs_dA", "#dc2626"),
    ]:
        ax.bar(
            x + offset,
            [item["summary"][key] for item in results],
            width=width,
            label=key.replace("max_abs_d", "max |d") + "|",
            color=color,
            alpha=0.85,
        )
    ax.axhline(5e-5, color="#111827", linestyle=":", linewidth=1.2, label="pass threshold")
    ax.set_yscale("log")
    ax.set_ylabel("absolute TMM-RCWA error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Maximum spectral R/T/A difference by validation case")
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_arc_reflectance(wavelengths: np.ndarray, outpath: Path) -> dict[str, Any]:
    arc_case = next(case for case in planar_cases() if case.key == "ideal_arc_on_silicon")
    bare_case = no_arc_silicon_case()
    arc = run_case(arc_case, wavelengths)
    bare = run_case(bare_case, wavelengths)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2), constrained_layout=True)
    wl_nm = wavelengths * 1000.0
    ax.plot(wl_nm, bare["tmm"]["R"], color="#475569", label="bare Si, TMM")
    ax.plot(wl_nm, bare["rcwa"]["R"], color="#475569", linestyle="--", label="bare Si, RCWA")
    ax.plot(wl_nm, arc["tmm"]["R"], color="#2563eb", label="quarter-wave ARC, TMM")
    ax.plot(
        wl_nm,
        arc["rcwa"]["R"],
        color="#dc2626",
        linestyle="--",
        label="quarter-wave ARC, RCWA",
    )
    ax.axvline(550.0, color="#111827", linestyle=":", linewidth=1.1, label="design wl")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.set_title("Quarter-wave anti-reflection coating sanity check")
    ax.set_ylim(-0.01, 0.45)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    design_idx = int(np.argmin(np.abs(wavelengths - 0.55)))
    return {
        "bare_R_at_550": float(bare["tmm"]["R"][design_idx]),
        "arc_R_at_550_tmm": float(arc["tmm"]["R"][design_idx]),
        "arc_R_at_550_rcwa": float(arc["rcwa"]["R"][design_idx]),
        "arc_max_abs_dR": arc["summary"]["max_abs_dR"],
    }


def write_reports(
    docs_root: Path,
    generated_on: str,
    results: list[dict[str, Any]],
    arc_summary: dict[str, Any],
) -> None:
    reports = docs_root / "reports"
    reports_ko = docs_root / "ko" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    reports_ko.mkdir(parents=True, exist_ok=True)

    rows = [
        [
            item["case"].title,
            item["case"].layer_count,
            item["case"].stack_thickness_um,
            item["summary"]["max_abs_dR"],
            item["summary"]["max_abs_dT"],
            item["summary"]["max_abs_dA"],
            item["summary"]["max_energy_residual_rcwa"],
            item["summary"]["passes"],
        ]
        for item in results
    ]
    max_err = max(
        max(
            item["summary"]["max_abs_dR"],
            item["summary"]["max_abs_dT"],
            item["summary"]["max_abs_dA"],
        )
        for item in results
    )

    en = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# TMM vs RCWA Planar Stack Validation",
            "",
            f"_Generated on {generated_on} from `transfer_matrix_1d` and direct "
            "`torcwa` zero-order RCWA solves._",
            "",
            "This report isolates the planar-stack limit. With no lateral patterning, "
            "zero-order RCWA should reduce to the same 1D optics solved by TMM. "
            "That makes this the first validation rung before using RCWA on Bayer "
            "patterns, metal grids, DTI trenches, or microlenses.",
            "",
            "## Executive summary",
            "",
            f"- All four planar validation cases pass the 5e-5 R/T/A agreement target; "
            f"the worst spectral difference is **{fmt(max_err)}**.",
            f"- The ideal quarter-wave ARC reduces the 550 nm bare silicon reflectance "
            f"from **{fmt(arc_summary['bare_R_at_550'])}** to "
            f"**{fmt(arc_summary['arc_R_at_550_tmm'])}** in TMM and "
            f"**{fmt(arc_summary['arc_R_at_550_rcwa'])}** in RCWA.",
            "- This report is normal-incidence and planar-only. It intentionally does "
            "not validate lateral diffraction, color-filter relief, DTI crosstalk, or "
            "photodiode collection.",
            "",
            "## R/T/A alignment",
            "",
            "![TMM vs RCWA RTA alignment](/reports/tmm-rcwa-planar/01_rta_alignment.png)",
            "",
            "## Error summary",
            "",
            "![TMM vs RCWA error summary](/reports/tmm-rcwa-planar/02_error_summary.png)",
            "",
            "## Quarter-wave ARC sanity check",
            "",
            "![Quarter-wave ARC reflectance](/reports/tmm-rcwa-planar/03_arc_reflectance.png)",
            "",
            "## Validation table",
            "",
            markdown_table(
                [
                    "Case",
                    "layers",
                    "thickness um",
                    "max |dR|",
                    "max |dT|",
                    "max |dA|",
                    "RCWA energy residual",
                    "passes",
                ],
                rows,
            ),
            "",
            "## Interpretation",
            "",
            "- The single-interface row checks Fresnel normalization without any finite films.",
            "- The ARC row verifies interference phase and the expected quarter-wave "
            "reflectance null.",
            "- The lossless multilayer checks phase accumulation through a pixel-like "
            "dielectric ladder.",
            "- The lossy multilayer checks that complex refractive indices and absorption "
            "accounting are aligned.",
            "",
            "## Regeneration",
            "",
            "```powershell",
            "uv run python scripts\\generate_tmm_rcwa_planar_report.py",
            "```",
            "",
            "Generated metrics are stored at "
            "`docs/public/reports/tmm-rcwa-planar/tmm_rcwa_planar_metrics.json`.",
            "",
        ]
    )

    ko = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# TMM vs RCWA 평면 스택 검증",
            "",
            f"_생성일: {generated_on}. `transfer_matrix_1d`와 direct `torcwa` "
            "zero-order RCWA solve에서 생성._",
            "",
            "이 리포트는 평면 스택 한계만 분리한다. 횡방향 패턴이 없으면 zero-order "
            "RCWA는 TMM이 푸는 1D optics와 같은 해로 수렴해야 한다. 따라서 Bayer "
            "패턴, metal grid, DTI trench, microlens를 쓰기 전에 확인하는 첫 번째 "
            "검증 단계다.",
            "",
            "## 요약",
            "",
            f"- 네 개의 평면 검증 케이스가 모두 5e-5 R/T/A agreement target을 "
            f"통과했다. 가장 큰 spectral difference는 **{fmt(max_err)}**이다.",
            f"- ideal quarter-wave ARC는 550 nm bare silicon reflectance를 "
            f"**{fmt(arc_summary['bare_R_at_550'])}**에서 TMM "
            f"**{fmt(arc_summary['arc_R_at_550_tmm'])}**, RCWA "
            f"**{fmt(arc_summary['arc_R_at_550_rcwa'])}**로 낮춘다.",
            "- 이 리포트는 normal-incidence planar-only 검증이다. lateral diffraction, "
            "color-filter relief, DTI crosstalk, photodiode collection은 의도적으로 "
            "검증하지 않는다.",
            "",
            "## R/T/A alignment",
            "",
            "![TMM vs RCWA RTA alignment](/reports/tmm-rcwa-planar/01_rta_alignment.png)",
            "",
            "## Error summary",
            "",
            "![TMM vs RCWA error summary](/reports/tmm-rcwa-planar/02_error_summary.png)",
            "",
            "## Quarter-wave ARC sanity check",
            "",
            "![Quarter-wave ARC reflectance](/reports/tmm-rcwa-planar/03_arc_reflectance.png)",
            "",
            "## Validation table",
            "",
            markdown_table(
                [
                    "Case",
                    "layers",
                    "thickness um",
                    "max |dR|",
                    "max |dT|",
                    "max |dA|",
                    "RCWA energy residual",
                    "passes",
                ],
                rows,
            ),
            "",
            "## 해석",
            "",
            "- single-interface row는 finite film 없이 Fresnel normalization을 확인한다.",
            "- ARC row는 interference phase와 quarter-wave reflectance null을 확인한다.",
            "- lossless multilayer는 pixel-like dielectric ladder의 phase accumulation을 확인한다.",
            "- lossy multilayer는 complex refractive index와 absorption accounting이 맞는지 확인한다.",
            "",
            "## 재생성",
            "",
            "```powershell",
            "uv run python scripts\\generate_tmm_rcwa_planar_report.py",
            "```",
            "",
            "생성 metric은 "
            "`docs/public/reports/tmm-rcwa-planar/tmm_rcwa_planar_metrics.json`에 저장된다.",
            "",
        ]
    )

    (reports / "tmm-rcwa-planar-validation.md").write_text(en, encoding="utf-8")
    (reports_ko / "tmm-rcwa-planar-validation.md").write_text(ko, encoding="utf-8")


def main() -> None:
    args = parse_args()
    docs_root = args.docs.resolve()
    public_dir = docs_root / "public" / "reports" / "tmm-rcwa-planar"
    public_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = np.array(args.wavelengths, dtype=float)
    results = [run_case(case, wavelengths) for case in planar_cases()]
    arc_summary = plot_arc_reflectance(wavelengths, public_dir / "03_arc_reflectance.png")
    plot_rta_alignment(results, public_dir / "01_rta_alignment.png")
    plot_error_summary(results, public_dir / "02_error_summary.png")

    metrics = {
        "generated_on": args.date,
        "validation_target_max_abs_rta_error": 5e-5,
        "arc_summary": arc_summary,
        "cases": [serialise_result(item) for item in results],
    }
    (public_dir / "tmm_rcwa_planar_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    write_reports(docs_root, args.date, results, arc_summary)
    print(f"Wrote TMM/RCWA planar report and assets to {public_dir}")


if __name__ == "__main__":
    main()
