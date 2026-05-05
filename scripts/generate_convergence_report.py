#!/usr/bin/env python3
"""Generate VitePress convergence report pages from local simulation outputs.

The benchmark scripts intentionally write large, ignored artifacts under
``outputs/``. This script promotes selected JSON metrics and PNG plots into the
documentation tree so the current convergence state can be reviewed through
GitHub Pages after the generated files are committed.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
PUBLIC_REPORT_DIR = DOCS / "public" / "reports" / "convergence"


@dataclass(frozen=True)
class ReportSource:
    key: str
    title: str
    source_dir: Path
    metric_file: str
    kind: str


ALIGNMENT_SOURCES = [
    ReportSource(
        "alignment_slab",
        "1D slab: TMM vs torcwa RCWA vs 1D FDTD",
        OUTPUTS / "rcwa_fdtd_alignment",
        "alignment_metrics.json",
        "alignment",
    ),
    ReportSource(
        "alignment_multilayer",
        "1D lossless multilayer: TMM vs torcwa RCWA vs 1D FDTD",
        OUTPUTS / "rcwa_fdtd_alignment_multilayer",
        "alignment_metrics.json",
        "alignment",
    ),
    ReportSource(
        "alignment_lossy_multilayer",
        "1D lossy pixel-like multilayer: TMM vs torcwa RCWA vs 1D FDTD",
        OUTPUTS / "rcwa_fdtd_alignment_lossy",
        "alignment_metrics.json",
        "alignment",
    ),
]

TRENCH_SOURCES = [
    ReportSource(
        "periodic_trench",
        "2D periodic FDTI/BDTI trench: torcwa RCWA vs 2D TE FDTD",
        OUTPUTS / "rcwa_fdtd_trench_benchmark",
        "trench_benchmark_metrics.json",
        "trench",
    ),
]

PIXEL_SOURCES = [
    ReportSource(
        "pixel_44x44x118_steps950",
        "2x2 pixel scalar FDTD, 44x44x118, 950 steps",
        OUTPUTS / "rcwa_fdtd_pixel_benchmark",
        "pixel_benchmark_metrics.json",
        "pixel",
    ),
    ReportSource(
        "pixel_44x44x118_steps2200",
        "2x2 pixel scalar FDTD, 44x44x118, 2200 steps",
        OUTPUTS / "rcwa_fdtd_pixel_benchmark_steps2200",
        "pixel_benchmark_metrics.json",
        "pixel",
    ),
    ReportSource(
        "pixel_44x44x118_steps3500",
        "2x2 pixel scalar FDTD, 44x44x118, 3500 steps",
        OUTPUTS / "rcwa_fdtd_pixel_benchmark_steps3500",
        "pixel_benchmark_metrics.json",
        "pixel",
    ),
    ReportSource(
        "pixel_64x64x170_steps3500",
        "2x2 pixel scalar FDTD, 64x64x170, 3500 steps",
        OUTPUTS / "rcwa_fdtd_pixel_benchmark_64x64x170_steps3500",
        "pixel_benchmark_metrics.json",
        "pixel",
    ),
    ReportSource(
        "pixel_64x64x170_steps5200",
        "2x2 pixel scalar FDTD, 64x64x170, 5200 steps",
        OUTPUTS / "rcwa_fdtd_pixel_benchmark_64x64x170_steps5200",
        "pixel_benchmark_metrics.json",
        "pixel",
    ),
    ReportSource(
        "pixel_128x128x340_single_steps10400",
        "2x2 pixel scalar FDTD, 128x128x340, single source, 10400 steps",
        OUTPUTS / "rcwa_fdtd_pixel_benchmark_128x128x340_single_steps10400",
        "pixel_benchmark_metrics.json",
        "pixel",
    ),
]

VISUAL_SOURCES = [
    ReportSource(
        "visual_cmos_pixel",
        "Visual CMOS pixel smoke test",
        OUTPUTS / "visual_cmos_optical_simulation",
        "visual_metrics.json",
        "visual",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate docs/reports pages from local convergence benchmark outputs.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=OUTPUTS,
        help="Directory containing ignored benchmark outputs.",
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=DOCS,
        help="VitePress docs directory.",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Report date to print in generated pages.",
    )
    return parser.parse_args()


def load_metric(source: ReportSource) -> dict[str, Any] | None:
    path = source.source_dir / source.metric_file
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
        if abs(value) >= 1000:
            return f"{value:.3g}"
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


def copy_pngs(source: ReportSource, docs_root: Path) -> list[dict[str, str]]:
    if not source.source_dir.exists():
        return []
    target_dir = docs_root / "public" / "reports" / "convergence" / source.key
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for image in sorted(source.source_dir.glob("*.png")):
        target = target_dir / image.name
        shutil.copy2(image, target)
        copied.append(
            {
                "title": image.stem.replace("_", " ").title(),
                "path": f"/reports/convergence/{source.key}/{image.name}",
            }
        )
    return copied


def image_block(images: list[dict[str, str]], limit: int | None = None) -> str:
    selected = images[:limit] if limit is not None else images
    if not selected:
        return "_No plots were found for this benchmark output._"
    lines: list[str] = []
    for image in selected:
        lines.append(f"![{image['title']}]({image['path']})")
        lines.append("")
        lines.append(f"*{image['title']}*")
        lines.append("")
    return "\n".join(lines).strip()


def first_metadata(metric: dict[str, Any], mode: str) -> dict[str, Any]:
    signals = metric.get("pd_signals", {}).get(mode, {})
    if not isinstance(signals, dict) or not signals:
        return {}
    first = next(iter(signals.values()))
    if isinstance(first, dict):
        metadata = first.get("metadata", {})
        if isinstance(metadata, dict):
            return metadata
    return {}


def source_status(source: ReportSource) -> str:
    if not source.source_dir.exists():
        return "missing directory"
    if not (source.source_dir / source.metric_file).exists():
        return "missing metrics"
    return "available"


def collect_sources(sources: list[ReportSource], docs_root: Path) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for source in sources:
        metric = load_metric(source)
        collected.append(
            {
                "source": source,
                "status": source_status(source),
                "metric": metric,
                "images": copy_pngs(source, docs_root),
            }
        )
    return collected


def alignment_rows(items: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in items:
        metric = item["metric"]
        source = item["source"]
        if metric is None:
            rows.append([source.title, item["status"], None, None, None, None])
            continue
        summary = metric.get("summary", {})
        errors = summary.get("max_abs_error_vs_tmm", {})
        status = summary.get("status", {})
        rows.append(
            [
                summary.get("structure_name", source.key),
                status.get("rcwa_torcwa_aligned_to_tmm"),
                status.get("fdtd_1d_aligned_to_tmm"),
                errors.get("rcwa_torcwa_R"),
                errors.get("fdtd_1d_R"),
                errors.get("fdtd_1d_T"),
            ]
        )
    return rows


def trench_rows(items: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in items:
        metric = item["metric"]
        if metric is None:
            continue
        summary = metric.get("summary", {})
        settings = metric.get("settings", {})
        for mode, payload in summary.items():
            errors = payload.get("max_abs_error_rcwa_vs_fdtd", {})
            rows.append(
                [
                    mode.upper(),
                    settings.get("rcwa_order"),
                    settings.get("fdtd_dx_um"),
                    errors.get("R"),
                    errors.get("T"),
                    errors.get("A"),
                    payload.get("mean_fdtd_trench_field_leakage"),
                    payload.get("status", {}).get("rta_aligned_for_periodic_trench"),
                ]
            )
    return rows


def pixel_rows(items: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for item in items:
        metric = item["metric"]
        source = item["source"]
        if metric is None:
            rows.append([source.title, item["status"], None, None, None, None, None, None])
            continue
        settings = metric.get("settings", {})
        grid = settings.get("fdtd_grid", [])
        source_set = settings.get("source_set")
        grid_label = "x".join(str(v) for v in grid) if grid else "-"
        for mode, payload in metric.get("summary", {}).items():
            metadata = first_metadata(metric, mode)
            physical_time_um = None
            if metadata.get("dt_um_over_c") is not None and metadata.get("steps") is not None:
                physical_time_um = metadata["dt_um_over_c"] * metadata["steps"]
            rows.append(
                [
                    grid_label,
                    settings.get("fdtd_steps"),
                    source_set,
                    mode.upper(),
                    metadata.get("dx_um"),
                    physical_time_um,
                    payload.get("mean_self_collection_fraction"),
                    payload.get("max_neighbor_crosstalk_fraction"),
                    payload.get("max_energy_tail_relative_change"),
                    len(payload.get("warnings", [])),
                ]
            )
    return rows


def pixel_plot_points(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for item in items:
        metric = item["metric"]
        if metric is None:
            continue
        settings = metric.get("settings", {})
        grid = settings.get("fdtd_grid", [])
        grid_label = "x".join(str(v) for v in grid) if grid else item["source"].key
        for mode, payload in metric.get("summary", {}).items():
            metadata = first_metadata(metric, mode)
            physical_time_um = None
            if metadata.get("dt_um_over_c") is not None and metadata.get("steps") is not None:
                physical_time_um = metadata["dt_um_over_c"] * metadata["steps"]
            points.append(
                {
                    "label": f"{grid_label}\n{settings.get('fdtd_steps')} steps",
                    "mode": mode.upper(),
                    "source_set": settings.get("source_set"),
                    "dx_um": metadata.get("dx_um"),
                    "physical_time_um": physical_time_um,
                    "self": payload.get("mean_self_collection_fraction"),
                    "xt": payload.get("max_neighbor_crosstalk_fraction"),
                    "tail": payload.get("max_energy_tail_relative_change"),
                }
            )
    return points


def write_pixel_summary_plot(items: list[dict[str, Any]], docs_root: Path) -> str | None:
    points = pixel_plot_points(items)
    if not points:
        return None

    outdir = docs_root / "public" / "reports" / "convergence"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "pixel_convergence_summary.png"

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    colors = {"FDTI": "#2563eb", "BDTI": "#dc2626"}
    markers = {"all": "o", "single": "s", "diagonal": "^"}

    for mode in ["FDTI", "BDTI"]:
        mode_points = [p for p in points if p["mode"] == mode]
        if not mode_points:
            continue
        x = np.arange(len(mode_points))
        tails = [p["tail"] for p in mode_points]
        self_values = [p["self"] for p in mode_points]
        xt_values = [p["xt"] for p in mode_points]
        axes[0].plot(x, tails, marker="o", color=colors[mode], label=mode)
        axes[1].plot(x, self_values, marker="o", color=colors[mode], label=f"{mode} self")
        axes[1].plot(
            x,
            xt_values,
            marker="x",
            linestyle="--",
            color=colors[mode],
            label=f"{mode} max neighbor",
        )
        for index, point in enumerate(mode_points):
            marker = markers.get(str(point["source_set"]), "o")
            axes[0].scatter(index, point["tail"], color=colors[mode], marker=marker, s=60)

    labels = [p["label"] for p in points if p["mode"] == "FDTI"] or [
        p["label"] for p in points
    ]
    axes[0].axhline(0.15, color="#64748b", linestyle=":", label="tail warning threshold")
    axes[0].set_title("FDTD energy-tail convergence")
    axes[0].set_ylabel("relative change in final 10% samples")
    axes[0].set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Photodiode collection fractions")
    axes[1].set_ylabel("fraction")
    axes[1].set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)

    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    return "/reports/convergence/pixel_convergence_summary.png"


def build_manifest(groups: dict[str, list[dict[str, Any]]], docs_root: Path) -> None:
    manifest = {
        group: [
            {
                "key": item["source"].key,
                "title": item["source"].title,
                "source_dir": str(item["source"].source_dir.relative_to(ROOT)),
                "metric_file": item["source"].metric_file,
                "status": item["status"],
                "images": item["images"],
            }
            for item in items
        ]
        for group, items in groups.items()
    }
    target = docs_root / "public" / "reports" / "convergence" / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_english_report(
    generated_on: str,
    groups: dict[str, list[dict[str, Any]]],
    pixel_summary_image: str | None,
) -> str:
    alignment = groups["alignment"]
    trench = groups["trench"]
    pixel = groups["pixel"]
    visual = groups["visual"]

    lines = [
        "---",
        "outline: deep",
        "---",
        "",
        "# RCWA/FDTD Convergence Analysis Report",
        "",
        f"_Generated on {generated_on} from local `outputs/` benchmark artifacts._",
        "",
        "This page turns the Python-level benchmark outputs into a publication-style report. "
        "The source JSON and plots stay in `outputs/` for local iteration, while selected "
        "figures and tables are promoted to `docs/public/reports/convergence/` so the "
        "same evidence can be served by GitHub Pages.",
        "",
        "## Current status",
        "",
        "- The 1D ladder is aligned: torcwa RCWA matches TMM at near numerical precision, "
        "and the 1D FDTD implementation is within the sub-percent target on the lossy "
        "pixel-like multilayer.",
        "- The 2D periodic trench benchmark is aligned for FDTI and BDTI at the current "
        "coarse settings, with maximum R/T/A differences below roughly 3 percentage "
        "points.",
        "- The full 2x2 pixel scalar FDTD runs are visual convergence tests. The "
        "44x44x118, 3500-step run is the current stable comparison point. The 64x64x170 "
        "and 128x128x340 runs need longer physical runtime before their crosstalk "
        "fractions should be treated as final.",
        "",
        "::: warning Read the high-resolution FDTD rows carefully",
        "A fixed `--fdtd-steps` value is not equivalent across grids. Finer grids use a "
        "smaller time step, so they cover less physical time unless the step count is "
        "scaled. Compare the reported `c*time` and energy-tail values, not only the "
        "grid dimensions.",
        ":::",
        "",
        "## Regeneration commands",
        "",
        "```powershell",
        "uv run python scripts\\rcwa_fdtd_alignment.py --structure lossy-multilayer --outdir outputs\\rcwa_fdtd_alignment_lossy",
        "uv run python scripts\\rcwa_fdtd_trench_benchmark.py --convergence --outdir outputs\\rcwa_fdtd_trench_benchmark",
        "uv run python scripts\\rcwa_fdtd_pixel_benchmark.py --fdtd-steps 3500 --outdir outputs\\rcwa_fdtd_pixel_benchmark_steps3500",
        "uv run python scripts\\generate_convergence_report.py",
        "```",
        "",
        "High-resolution pixel checks can be regenerated with:",
        "",
        "```powershell",
        "uv run python scripts\\rcwa_fdtd_pixel_benchmark.py --nx 64 --ny 64 --nz 170 --fdtd-steps 5200 --outdir outputs\\rcwa_fdtd_pixel_benchmark_64x64x170_steps5200",
        "uv run python scripts\\rcwa_fdtd_pixel_benchmark.py --nx 128 --ny 128 --nz 340 --source-set single --fdtd-steps 10400 --outdir outputs\\rcwa_fdtd_pixel_benchmark_128x128x340_single_steps10400",
        "uv run python scripts\\generate_convergence_report.py",
        "```",
        "",
        "## 1D solver-alignment ladder",
        "",
        markdown_table(
            [
                "Structure",
                "RCWA ok",
                "FDTD ok",
                "max |Rrcwa-Rtmm|",
                "max |Rfdtd-Rtmm|",
                "max |Tfdtd-Ttmm|",
            ],
            alignment_rows(alignment),
        ),
        "",
    ]

    for item in alignment:
        source = item["source"]
        lines.extend(
            [
                f"### {source.title}",
                "",
                image_block(item["images"], limit=3),
                "",
            ]
        )

    lines.extend(
        [
            "## 2D FDTI/BDTI periodic trench",
            "",
            "This benchmark uses one shared periodic trench geometry for both solvers. "
            "It is the first rung where FDTI and BDTI directionality matters.",
            "",
            markdown_table(
                [
                    "Mode",
                    "RCWA order",
                    "FDTD dx um",
                    "max |dR|",
                    "max |dT|",
                    "max |dA|",
                    "field leakage",
                    "aligned",
                ],
                trench_rows(trench),
            ),
            "",
        ]
    )

    for item in trench:
        lines.extend(
            [
                f"### {item['source'].title}",
                "",
                image_block(item["images"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Full 2x2 Bayer pixel convergence",
            "",
            "The pixel benchmark uses the real `PixelStack` path with FDTI/BDTI options, "
            "material-database complex indices, BARL layers, microlens slices, color "
            "filters, and photodiode integration windows. RCWA reports full-supercell "
            "R/T/A, while the scalar FDTD path reports localized-source collection and "
            "crosstalk proxies.",
            "",
        ]
    )

    if pixel_summary_image:
        lines.extend(
            [
                f"![Pixel convergence summary]({pixel_summary_image})",
                "",
                "*Pixel convergence summary generated from all available pixel benchmark metrics.*",
                "",
            ]
        )

    lines.extend(
        [
            markdown_table(
                [
                    "Grid",
                    "steps",
                    "sources",
                    "mode",
                    "dx um",
                    "c*time um",
                    "self frac",
                    "max neighbor",
                    "tail",
                    "warnings",
                ],
                pixel_rows(pixel),
            ),
            "",
            "### Pixel plots",
            "",
            "The images below are copied from the benchmark output folders. Use the geometry "
            "and field-slice plots to catch direction, indexing, and source-placement "
            "issues that scalar metrics alone can hide.",
            "",
        ]
    )

    for item in pixel:
        if item["metric"] is None:
            continue
        lines.extend(
            [
                f"#### {item['source'].title}",
                "",
                image_block(item["images"], limit=4),
                "",
            ]
        )

    lines.extend(
        [
            "## Visual smoke-test artifacts",
            "",
            "These plots are not used as rigorous solver evidence. They are retained as "
            "fast visual tests for the FDTI/BDTI geometry, photodiode windows, and "
            "plotting pipeline.",
            "",
        ]
    )

    for item in visual:
        if item["metric"] is None:
            continue
        lines.extend(
            [
                f"### {item['source'].title}",
                "",
                image_block(item["images"]),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Use the 1D ladder to validate normalization, material loss, and monitor "
            "math before debugging full pixels.",
            "- Use the periodic trench benchmark to compare FDTI and BDTI with the same "
            "geometry, boundary conditions, and R/T/A definitions.",
            "- Treat full-pixel scalar FDTD as a visual convergence and crosstalk proxy "
            "until the energy tail falls below the selected threshold at the target grid.",
            "- For final high-accuracy work, scale FDTD steps with grid refinement, run "
            "all four sources, and repeat the RCWA side with Fourier order and "
            "permittivity-grid sweeps.",
            "",
        ]
    )
    return "\n".join(lines)


def build_korean_report(
    generated_on: str,
    groups: dict[str, list[dict[str, Any]]],
    pixel_summary_image: str | None,
) -> str:
    lines = [
        "---",
        "outline: deep",
        "---",
        "",
        "# RCWA/FDTD 수렴 분석 리포트",
        "",
        f"_생성일: {generated_on}. 로컬 `outputs/` 벤치마크 산출물에서 생성됨._",
        "",
        "이 페이지는 Python 레벨 벤치마크 결과를 GitHub Pages에서 볼 수 있는 "
        "리포트 형태로 정리한다. 큰 원본 산출물은 `outputs/`에 두고, 선별된 "
        "그림과 표만 `docs/public/reports/convergence/`로 복사한다.",
        "",
        "## 현재 판단",
        "",
        "- 1D 정합성 사다리는 통과했다. torcwa RCWA는 TMM과 수치 오차 수준으로 "
        "맞고, 1D FDTD도 lossy pixel-like multilayer에서 sub-percent 범위다.",
        "- 2D periodic trench는 현재 coarse 설정에서 FDTI/BDTI 모두 R/T/A 차이가 "
        "대략 3 percentage point 이하로 맞는다.",
        "- 전체 2x2 pixel scalar FDTD는 아직 최종 물리 결과가 아니라 시각적 "
        "수렴 테스트다. 현재 안정적으로 볼 기준점은 44x44x118, 3500 steps이고, "
        "64x64x170 및 128x128x340 결과는 더 긴 물리 시간이 필요하다.",
        "",
        "::: warning 고해상도 FDTD 행 해석 주의",
        "`--fdtd-steps`를 고정하면 격자가 촘촘해질수록 실제 물리 시간이 짧아진다. "
        "고해상도 행은 grid 크기만 보지 말고 `c*time`과 energy tail 값을 함께 봐야 한다.",
        ":::",
        "",
        "## 재생성 명령",
        "",
        "```powershell",
        "uv run python scripts\\rcwa_fdtd_alignment.py --structure lossy-multilayer --outdir outputs\\rcwa_fdtd_alignment_lossy",
        "uv run python scripts\\rcwa_fdtd_trench_benchmark.py --convergence --outdir outputs\\rcwa_fdtd_trench_benchmark",
        "uv run python scripts\\rcwa_fdtd_pixel_benchmark.py --fdtd-steps 3500 --outdir outputs\\rcwa_fdtd_pixel_benchmark_steps3500",
        "uv run python scripts\\generate_convergence_report.py",
        "```",
        "",
        "고해상도 pixel 체크는 다음 명령으로 다시 만들 수 있다.",
        "",
        "```powershell",
        "uv run python scripts\\rcwa_fdtd_pixel_benchmark.py --nx 64 --ny 64 --nz 170 --fdtd-steps 5200 --outdir outputs\\rcwa_fdtd_pixel_benchmark_64x64x170_steps5200",
        "uv run python scripts\\rcwa_fdtd_pixel_benchmark.py --nx 128 --ny 128 --nz 340 --source-set single --fdtd-steps 10400 --outdir outputs\\rcwa_fdtd_pixel_benchmark_128x128x340_single_steps10400",
        "uv run python scripts\\generate_convergence_report.py",
        "```",
        "",
        "## 1D 솔버 정합성 사다리",
        "",
        markdown_table(
            [
                "구조",
                "RCWA 통과",
                "FDTD 통과",
                "max |Rrcwa-Rtmm|",
                "max |Rfdtd-Rtmm|",
                "max |Tfdtd-Ttmm|",
            ],
            alignment_rows(groups["alignment"]),
        ),
        "",
    ]

    for item in groups["alignment"]:
        lines.extend(
            [
                f"### {item['source'].title}",
                "",
                image_block(item["images"], limit=3),
                "",
            ]
        )

    lines.extend(
        [
            "## 2D FDTI/BDTI periodic trench",
            "",
            "FDTI와 BDTI 방향성이 처음으로 실제 오차에 영향을 주는 단계다. "
            "두 솔버가 같은 periodic trench 구조와 같은 R/T/A 정의를 사용한다.",
            "",
            markdown_table(
                [
                    "모드",
                    "RCWA order",
                    "FDTD dx um",
                    "max |dR|",
                    "max |dT|",
                    "max |dA|",
                    "field leakage",
                    "정합",
                ],
                trench_rows(groups["trench"]),
            ),
            "",
        ]
    )

    for item in groups["trench"]:
        lines.extend(
            [
                f"### {item['source'].title}",
                "",
                image_block(item["images"]),
                "",
            ]
        )

    lines.extend(
        [
            "## 전체 2x2 Bayer pixel 수렴",
            "",
            "Pixel benchmark는 실제 `PixelStack` 경로를 사용한다. FDTI/BDTI 옵션, "
            "복소 굴절률 재료, BARL, microlens slice, color filter, photodiode "
            "integration window가 모두 들어간다. RCWA는 full-supercell R/T/A를, "
            "scalar FDTD는 localized-source collection 및 crosstalk proxy를 보고한다.",
            "",
        ]
    )

    if pixel_summary_image:
        lines.extend(
            [
                f"![Pixel convergence summary]({pixel_summary_image})",
                "",
                "*사용 가능한 모든 pixel benchmark metric에서 생성한 수렴 요약.*",
                "",
            ]
        )

    lines.extend(
        [
            markdown_table(
                [
                    "Grid",
                    "steps",
                    "sources",
                    "mode",
                    "dx um",
                    "c*time um",
                    "self frac",
                    "max neighbor",
                    "tail",
                    "warnings",
                ],
                pixel_rows(groups["pixel"]),
            ),
            "",
            "### Pixel plots",
            "",
            "아래 이미지는 benchmark output 폴더에서 복사한 것이다. geometry와 field slice "
            "그림은 숫자 metric만으로 놓치기 쉬운 방향, indexing, source-placement 문제를 "
            "확인하는 데 중요하다.",
            "",
        ]
    )

    for item in groups["pixel"]:
        if item["metric"] is None:
            continue
        lines.extend(
            [
                f"#### {item['source'].title}",
                "",
                image_block(item["images"], limit=4),
                "",
            ]
        )

    lines.extend(
        [
            "## 시각적 smoke test",
            "",
            "이 그림들은 엄밀한 solver evidence가 아니다. FDTI/BDTI geometry, photodiode "
            "window, plotting pipeline을 빠르게 눈으로 확인하기 위한 테스트로 유지한다.",
            "",
        ]
    )

    for item in groups["visual"]:
        if item["metric"] is None:
            continue
        lines.extend(
            [
                f"### {item['source'].title}",
                "",
                image_block(item["images"]),
                "",
            ]
        )

    lines.extend(
        [
            "## 해석 기준",
            "",
            "- 전체 pixel을 보기 전에 1D 사다리로 normalization, material loss, monitor "
            "계산을 먼저 검증한다.",
            "- FDTI/BDTI 비교는 periodic trench benchmark에서 같은 geometry와 R/T/A "
            "정의로 먼저 맞춘다.",
            "- full-pixel scalar FDTD는 목표 grid에서 energy tail이 기준 이하로 내려갈 "
            "때까지 시각적 수렴 및 crosstalk proxy로 해석한다.",
            "- 최종 고정밀 평가는 FDTD step을 grid refinement에 맞춰 늘리고, 네 source를 "
            "모두 돌리며, RCWA도 Fourier order와 permittivity grid sweep을 같이 수행해야 한다.",
            "",
        ]
    )
    return "\n".join(lines)


def write_index_pages(docs_root: Path, generated_on: str) -> None:
    reports = docs_root / "reports"
    reports_ko = docs_root / "ko" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    reports_ko.mkdir(parents=True, exist_ok=True)

    english = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# Simulation Reports",
            "",
            "Publication-style reports generated from Python benchmark artifacts.",
            "",
            f"- [RCWA/FDTD Convergence Analysis](./convergence-analysis.md) "
            f"(generated {generated_on})",
            "",
            "The report assets are served from `docs/public/reports/`, so they are "
            "included in the VitePress build and the GitHub Pages deployment.",
            "",
        ]
    )
    korean = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# 시뮬레이션 리포트",
            "",
            "Python benchmark 산출물에서 생성한 publication-style 리포트다.",
            "",
            f"- [RCWA/FDTD 수렴 분석](./convergence-analysis.md) "
            f"(생성일 {generated_on})",
            "",
            "리포트 이미지는 `docs/public/reports/`에서 서빙되므로 VitePress build와 "
            "GitHub Pages 배포에 포함된다.",
            "",
        ]
    )
    (reports / "index.md").write_text(english, encoding="utf-8")
    (reports_ko / "index.md").write_text(korean, encoding="utf-8")


def main() -> None:
    args = parse_args()
    global OUTPUTS, DOCS, PUBLIC_REPORT_DIR
    OUTPUTS = args.outputs.resolve()
    DOCS = args.docs.resolve()
    PUBLIC_REPORT_DIR = DOCS / "public" / "reports" / "convergence"

    # Rebase source paths if non-default directories are supplied.
    def rebase(source: ReportSource) -> ReportSource:
        try:
            relative = source.source_dir.relative_to(ROOT / "outputs")
        except ValueError:
            relative = source.source_dir.name
        return ReportSource(
            source.key,
            source.title,
            OUTPUTS / relative,
            source.metric_file,
            source.kind,
        )

    groups = {
        "alignment": collect_sources([rebase(s) for s in ALIGNMENT_SOURCES], DOCS),
        "trench": collect_sources([rebase(s) for s in TRENCH_SOURCES], DOCS),
        "pixel": collect_sources([rebase(s) for s in PIXEL_SOURCES], DOCS),
        "visual": collect_sources([rebase(s) for s in VISUAL_SOURCES], DOCS),
    }

    pixel_summary = write_pixel_summary_plot(groups["pixel"], DOCS)
    build_manifest(groups, DOCS)
    write_index_pages(DOCS, args.date)

    reports = DOCS / "reports"
    reports_ko = DOCS / "ko" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    reports_ko.mkdir(parents=True, exist_ok=True)

    (reports / "convergence-analysis.md").write_text(
        build_english_report(args.date, groups, pixel_summary),
        encoding="utf-8",
    )
    (reports_ko / "convergence-analysis.md").write_text(
        build_korean_report(args.date, groups, pixel_summary),
        encoding="utf-8",
    )

    print(f"Wrote reports to {reports} and {reports_ko}")
    print(f"Wrote public assets to {PUBLIC_REPORT_DIR}")


if __name__ == "__main__":
    main()
