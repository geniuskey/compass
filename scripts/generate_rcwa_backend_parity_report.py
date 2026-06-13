#!/usr/bin/env python3
"""Generate the RCWA backend parity report.

The report runs the same small PixelStack through every RCWA adapter registered
in COMPASS and publishes the result as a backend-health gate. It is intentionally
small enough to regenerate in CI or before a release, while still using the real
PixelStack -> solver adapter path for patterned CMOS pixels.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import warnings
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
from compass.solvers.base import SolverFactory  # noqa: E402

DOCS = ROOT / "docs"
PUBLIC_DIR = DOCS / "public" / "reports" / "rcwa-backend-parity"


@dataclass(frozen=True)
class BackendSpec:
    name: str
    label: str
    config: dict[str, Any]
    role: str


BACKENDS = [
    BackendSpec(
        name="torcwa",
        label="torcwa",
        role="reference",
        config={
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
        },
    ),
    BackendSpec(
        name="grcwa",
        label="grcwa",
        role="candidate",
        config={
            "name": "grcwa",
            "type": "rcwa",
            "params": {
                "fourier_order": [3, 3],
                "n_lens_slices": 4,
                "grid_multiplier": 2,
            },
        },
    ),
    BackendSpec(
        name="meent",
        label="meent/numpy",
        role="candidate",
        config={
            "name": "meent",
            "type": "rcwa",
            "params": {
                "fourier_order": [1, 1],
                "backend": "numpy",
            },
        },
    ),
    BackendSpec(
        name="fmmax",
        label="fmmax",
        role="candidate",
        config={
            "name": "fmmax",
            "type": "rcwa",
            "params": {
                "fourier_order": [3, 3],
                "fmm_formulation": "jones",
                "dtype": "complex64",
            },
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RCWA backend parity report.")
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=[0.45, 0.55, 0.65],
        help="Wavelengths in um for the parity smoke run.",
    )
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


def source_config(wavelengths: np.ndarray) -> dict[str, Any]:
    return {
        "wavelength": {"mode": "list", "values": [float(wl) for wl in wavelengths]},
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "TE",
    }


def average_qe(qe_per_pixel: dict[str, np.ndarray], n_wl: int) -> np.ndarray:
    if not qe_per_pixel:
        return np.zeros(n_wl)
    arrays = [np.asarray(values, dtype=float) for values in qe_per_pixel.values()]
    return np.mean(np.vstack(arrays), axis=0)


def run_backend(
    spec: BackendSpec,
    stack: PixelStack,
    wavelengths: np.ndarray,
) -> dict[str, Any]:
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(min(old_level, logging.WARNING) if old_level else logging.WARNING)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            solver = SolverFactory.create(spec.name, spec.config, "cpu")
            solver.setup_geometry(stack)
            solver.setup_source(source_config(wavelengths))
            result = solver.run_timed()
            reflection = (
                np.asarray(result.reflection, dtype=float)
                if result.reflection is not None
                else np.full(len(wavelengths), np.nan)
            )
            transmission = (
                np.asarray(result.transmission, dtype=float)
                if result.transmission is not None
                else np.full(len(wavelengths), np.nan)
            )
            absorption = (
                np.asarray(result.absorption, dtype=float)
                if result.absorption is not None
                else np.full(len(wavelengths), np.nan)
            )
            qe_mean = average_qe(result.qe_per_pixel, len(wavelengths))
            metadata = dict(result.metadata)
            exception = None
        except Exception as exc:  # pragma: no cover - report must capture backend failures.
            reflection = np.full(len(wavelengths), np.nan)
            transmission = np.full(len(wavelengths), np.nan)
            absorption = np.full(len(wavelengths), np.nan)
            qe_mean = np.full(len(wavelengths), np.nan)
            metadata = {}
            exception = repr(exc)
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)

    finite = (
        np.all(np.isfinite(reflection))
        and np.all(np.isfinite(transmission))
        and np.all(np.isfinite(absorption))
    )
    energy = reflection + transmission + absorption
    energy_residual = float(np.nanmax(np.abs(energy - 1.0))) if finite else math.nan
    bounded = bool(
        finite
        and np.nanmin([reflection.min(), transmission.min(), absorption.min()]) >= -0.05
        and np.nanmax([reflection.max(), transmission.max(), absorption.max()]) <= 1.05
    )
    not_all_zero = bool(
        finite
        and np.nanmax(np.abs(reflection) + np.abs(transmission) + np.abs(absorption)) > 1e-8
    )
    physical_ok = bool(bounded and energy_residual <= 0.05 and not_all_zero)
    warnings_text = [str(item.message) for item in caught]
    logs = [line for line in log_stream.getvalue().splitlines() if line.strip()]

    return {
        "name": spec.name,
        "label": spec.label,
        "role": spec.role,
        "config": spec.config,
        "exception": exception,
        "warnings": warnings_text,
        "logs": logs,
        "reflection": reflection,
        "transmission": transmission,
        "absorption": absorption,
        "qe_mean": qe_mean,
        "runtime_s": float(metadata.get("runtime_seconds", math.nan)),
        "energy_residual": energy_residual,
        "physical_ok": physical_ok,
        "bounded": bounded,
        "metadata": metadata,
    }


def classify(results: list[dict[str, Any]]) -> None:
    reference = next((item for item in results if item["name"] == "torcwa"), None)
    if reference is None or not reference["physical_ok"]:
        for item in results:
            item["max_abs_rta_delta_vs_torcwa"] = math.nan
            item["parity_ok"] = False
        return

    ref_arrays = {
        "R": reference["reflection"],
        "T": reference["transmission"],
        "A": reference["absorption"],
    }
    for item in results:
        if not item["physical_ok"]:
            item["max_abs_rta_delta_vs_torcwa"] = math.nan
            item["parity_ok"] = False
            continue
        deltas = [
            np.nanmax(np.abs(item["reflection"] - ref_arrays["R"])),
            np.nanmax(np.abs(item["transmission"] - ref_arrays["T"])),
            np.nanmax(np.abs(item["absorption"] - ref_arrays["A"])),
        ]
        max_delta = float(np.max(deltas))
        item["max_abs_rta_delta_vs_torcwa"] = max_delta
        item["parity_ok"] = bool(item["name"] == "torcwa" or max_delta <= 0.05)


def serialise_result(item: dict[str, Any], wavelengths: np.ndarray) -> dict[str, Any]:
    return {
        "name": item["name"],
        "label": item["label"],
        "role": item["role"],
        "physical_ok": item["physical_ok"],
        "parity_ok": item["parity_ok"],
        "bounded": item["bounded"],
        "runtime_s": json_safe(item["runtime_s"]),
        "energy_residual": json_safe(item["energy_residual"]),
        "max_abs_rta_delta_vs_torcwa": json_safe(item["max_abs_rta_delta_vs_torcwa"]),
        "exception": item["exception"],
        "warnings": item["warnings"],
        "logs": item["logs"],
        "metadata": json_safe(item["metadata"]),
        "spectra": {
            "wavelength_um": json_safe(wavelengths),
            "reflection": json_safe(item["reflection"]),
            "transmission": json_safe(item["transmission"]),
            "absorption": json_safe(item["absorption"]),
            "qe_mean": json_safe(item["qe_mean"]),
        },
    }


def plot_spectra(results: list[dict[str, Any]], wavelengths: np.ndarray, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True, constrained_layout=True)
    colors = {
        "torcwa": "#2563eb",
        "grcwa": "#16a34a",
        "meent": "#f97316",
        "fmmax": "#dc2626",
    }
    wl_nm = wavelengths * 1000.0
    for ax, key, title in zip(
        axes,
        ["reflection", "transmission", "absorption"],
        ["Reflection", "Transmission", "Absorption"],
    ):
        for item in results:
            values = item[key]
            clipped = np.clip(values, -0.02, 1.08)
            label = item["label"]
            if np.any(np.isfinite(values) & ((values < -0.02) | (values > 1.08))):
                label += " (clipped)"
            ax.plot(
                wl_nm,
                clipped,
                marker="o",
                linewidth=1.8,
                label=label,
                color=colors.get(item["name"]),
            )
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlabel("wavelength (nm)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("R/T/A value (clipped to plotting range)")
    axes[2].legend(fontsize=8, loc="best")
    fig.suptitle("RCWA backend parity smoke: same PixelStack, low-order TE run", fontsize=15)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_health(results: list[dict[str, Any]], outpath: Path) -> None:
    labels = [item["label"] for item in results]
    x = np.arange(len(labels))
    colors = ["#2563eb" if item["parity_ok"] else "#dc2626" for item in results]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    deltas = [
        item["max_abs_rta_delta_vs_torcwa"]
        if math.isfinite(item["max_abs_rta_delta_vs_torcwa"])
        else 10.0
        for item in results
    ]
    axes[0].bar(x, deltas, color=colors, alpha=0.86)
    axes[0].axhline(0.05, color="#111827", linestyle=":", label="parity target")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("max |R/T/A delta| vs torcwa")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].grid(True, axis="y", which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    runtimes = [item["runtime_s"] if math.isfinite(item["runtime_s"]) else 0.0 for item in results]
    axes[1].bar(x, runtimes, color="#475569", alpha=0.86)
    axes[1].set_ylabel("runtime (s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].set_title("Adapter runtime for 3-wavelength smoke")
    fig.suptitle("Backend health: parity target and runtime", fontsize=15)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def report_rows(results: list[dict[str, Any]], wavelengths: np.ndarray) -> list[list[Any]]:
    mid_idx = int(np.argmin(np.abs(wavelengths - 0.55)))
    rows: list[list[Any]] = []
    for item in results:
        first_log = item["exception"] or (item["logs"][0] if item["logs"] else "")
        if len(first_log) > 80:
            first_log = first_log[:77] + "..."
        rows.append(
            [
                item["label"],
                item["physical_ok"],
                item["parity_ok"],
                item["reflection"][mid_idx],
                item["transmission"][mid_idx],
                item["absorption"][mid_idx],
                item["qe_mean"][mid_idx],
                item["energy_residual"],
                item["max_abs_rta_delta_vs_torcwa"],
                item["runtime_s"],
                first_log or "-",
            ]
        )
    return rows


def write_reports(
    docs_root: Path,
    generated_on: str,
    results: list[dict[str, Any]],
    wavelengths: np.ndarray,
) -> None:
    reports = docs_root / "reports"
    reports_ko = docs_root / "ko" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    reports_ko.mkdir(parents=True, exist_ok=True)

    passing = [item for item in results if item["parity_ok"]]
    physical = [item for item in results if item["physical_ok"]]
    rows = report_rows(results, wavelengths)

    en = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# RCWA Backend Parity Report",
            "",
            f"_Generated on {generated_on} from the real `PixelStack` -> RCWA adapter path._",
            "",
            "This report runs a small 1.0 um BSI PixelStack through each RCWA backend "
            "adapter with the same TE source and wavelength list. It is a backend-health "
            "gate, not a converged optical benchmark: the Fourier orders are deliberately "
            "small so adapter failures and normalization mismatches are visible quickly.",
            "",
            "## Executive summary",
            "",
            f"- {len(physical)} of {len(results)} RCWA adapters produced bounded, "
            "energy-consistent R/T/A values in this smoke test.",
            f"- {len(passing)} of {len(results)} adapters met the parity target against "
            "`torcwa` (max R/T/A delta <= 0.05).",
            "- Current result: `torcwa` is the usable reference path for patterned "
            "PixelStack runs; the other adapters need normalization/API work before they "
            "should be used for production pixel comparisons.",
            "",
            "::: warning Low-order smoke test",
            "This report does not claim converged QE. It checks whether adapters can run "
            "the same patterned stack and return plausible R/T/A numbers. Converged "
            "solver settings belong in the convergence report.",
            ":::",
            "",
            "## R/T/A spectra",
            "",
            "![RCWA backend spectra](/reports/rcwa-backend-parity/01_backend_rta_spectra.png)",
            "",
            "## Health and runtime",
            "",
            "![RCWA backend health](/reports/rcwa-backend-parity/02_backend_health_runtime.png)",
            "",
            "## Backend table",
            "",
            markdown_table(
                [
                    "Backend",
                    "physical ok",
                    "parity ok",
                    "R@550",
                    "T@550",
                    "A@550",
                    "mean QE@550",
                    "energy residual",
                    "max delta vs torcwa",
                    "runtime s",
                    "first warning/error",
                ],
                rows,
            ),
            "",
            "## Interpretation",
            "",
            "- `physical ok` means finite, bounded R/T/A values and |R+T+A-1| <= 0.05.",
            "- `parity ok` means physical output and max R/T/A delta against `torcwa` <= 0.05.",
            "- `mean QE` is the adapter's photodiode-allocation proxy averaged over the "
            "four Bayer pixels. It is useful for detecting broken absorption allocation, "
            "not for design decisions at this low order.",
            "- A backend can import successfully but still fail this report if the adapter "
            "uses stale upstream APIs or incompatible normalization.",
            "",
            "## Regeneration",
            "",
            "```powershell",
            "uv run python scripts\\generate_rcwa_backend_parity_report.py",
            "```",
            "",
            "Generated metrics are stored at "
            "`docs/public/reports/rcwa-backend-parity/rcwa_backend_parity_metrics.json`.",
            "",
        ]
    )

    ko = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# RCWA Backend Parity 리포트",
            "",
            f"_생성일: {generated_on}. 실제 `PixelStack` -> RCWA adapter 경로에서 생성._",
            "",
            "이 리포트는 작은 1.0 um BSI PixelStack을 동일한 TE source와 wavelength "
            "list로 각 RCWA backend adapter에 실행한다. 목적은 backend-health gate다. "
            "Fourier order는 의도적으로 낮게 잡아 adapter failure와 normalization mismatch가 "
            "빨리 보이게 했다.",
            "",
            "## 요약",
            "",
            f"- {len(results)}개 RCWA adapter 중 {len(physical)}개가 bounded 및 "
            "energy-consistent R/T/A 값을 냈다.",
            f"- {len(results)}개 중 {len(passing)}개가 `torcwa` 대비 parity target "
            "(max R/T/A delta <= 0.05)을 만족했다.",
            "- 현재 결과에서 patterned PixelStack run의 usable reference path는 "
            "`torcwa`다. 나머지 adapter는 production pixel comparison 전에 "
            "normalization/API 보수가 필요하다.",
            "",
            "::: warning Low-order smoke test",
            "이 리포트는 converged QE를 주장하지 않는다. adapter가 같은 patterned stack을 "
            "실행하고 그럴듯한 R/T/A 값을 반환하는지 확인한다. 수렴 설정은 convergence "
            "report에서 다룬다.",
            ":::",
            "",
            "## R/T/A spectra",
            "",
            "![RCWA backend spectra](/reports/rcwa-backend-parity/01_backend_rta_spectra.png)",
            "",
            "## Health and runtime",
            "",
            "![RCWA backend health](/reports/rcwa-backend-parity/02_backend_health_runtime.png)",
            "",
            "## Backend table",
            "",
            markdown_table(
                [
                    "Backend",
                    "physical ok",
                    "parity ok",
                    "R@550",
                    "T@550",
                    "A@550",
                    "mean QE@550",
                    "energy residual",
                    "max delta vs torcwa",
                    "runtime s",
                    "first warning/error",
                ],
                rows,
            ),
            "",
            "## 해석",
            "",
            "- `physical ok`는 finite, bounded R/T/A 값과 |R+T+A-1| <= 0.05를 의미한다.",
            "- `parity ok`는 physical output이면서 `torcwa` 대비 max R/T/A delta <= 0.05임을 의미한다.",
            "- `mean QE`는 네 Bayer pixel에 대한 adapter의 photodiode-allocation proxy 평균이다. "
            "이 낮은 order에서는 설계 판단이 아니라 broken absorption allocation 감지용이다.",
            "- backend import가 성공해도 upstream API 변화나 normalization mismatch가 있으면 "
            "이 리포트에서 실패할 수 있다.",
            "",
            "## 재생성",
            "",
            "```powershell",
            "uv run python scripts\\generate_rcwa_backend_parity_report.py",
            "```",
            "",
            "생성 metric은 "
            "`docs/public/reports/rcwa-backend-parity/rcwa_backend_parity_metrics.json`에 저장된다.",
            "",
        ]
    )

    (reports / "rcwa-backend-parity.md").write_text(en, encoding="utf-8")
    (reports_ko / "rcwa-backend-parity.md").write_text(ko, encoding="utf-8")


def main() -> None:
    args = parse_args()
    docs_root = args.docs.resolve()
    public_dir = docs_root / "public" / "reports" / "rcwa-backend-parity"
    public_dir.mkdir(parents=True, exist_ok=True)

    wavelengths = np.array(args.wavelengths, dtype=float)
    stack = PixelStack({"pixel": derive_parameters("generic_bsi", pitch=1.0)})
    results = [run_backend(spec, stack, wavelengths) for spec in BACKENDS]
    classify(results)

    plot_spectra(results, wavelengths, public_dir / "01_backend_rta_spectra.png")
    plot_health(results, public_dir / "02_backend_health_runtime.png")

    metrics = {
        "generated_on": args.date,
        "benchmark": {
            "pixel": "generic_bsi",
            "pitch_um": 1.0,
            "wavelength_um": wavelengths.tolist(),
            "polarization": "TE",
            "angle_deg": 0.0,
            "parity_target_max_abs_rta_delta": 0.05,
        },
        "backends": [serialise_result(item, wavelengths) for item in results],
    }
    (public_dir / "rcwa_backend_parity_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_reports(docs_root, args.date, results, wavelengths)
    print(f"Wrote RCWA backend parity report and assets to {public_dir}")


if __name__ == "__main__":
    main()
