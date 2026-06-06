#!/usr/bin/env python3
"""Generate the "Pixel Structure Realism" VitePress report.

This report documents the structural-realism features COMPASS models for CMOS
image-sensor pixels and visualises them directly from the generated solver
input (the actual `PixelStack` layer slices), not from schematics. It compares a
baseline pixel against a realism-enhanced (NIR) pixel that uses a backside
inverted-pyramid texture, a tapered DTI trench with a conformal high-k liner,
and a microlens residual base layer.

The figures sample the real permittivity eps(x, z) the RCWA solver receives, so
the report is geometry evidence rather than an optical-performance claim.
"""

from __future__ import annotations

import argparse
import json
import sys
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

DOCS = ROOT / "docs"
PUBLIC_DIR = DOCS / "public" / "reports" / "geometry" / "structure-realism"

# Capability coverage matrix: real-pixel feature -> COMPASS modelling status.
# "config" lists the config knob(s) that activate the feature.
COVERAGE: list[dict[str, str]] = [
    {
        "feature": "Microlens superellipse profile + CRA shift",
        "status": "modelled",
        "config": "microlens.profile, microlens.shift",
    },
    {
        "feature": "Multi-pixel shared lens (2x2 / 4x4 OCL)",
        "status": "modelled",
        "config": "microlens.sharing",
    },
    {
        "feature": "Microlens residual base layer",
        "status": "modelled (new)",
        "config": "microlens.base_thickness",
    },
    {
        "feature": "Per-color CF thickness + contact-angle relief",
        "status": "modelled",
        "config": "color_filter.<color>.thickness/contact_angle",
    },
    {
        "feature": "Metal grid (W) with rounded corners",
        "status": "modelled",
        "config": "color_filter.grid",
    },
    {
        "feature": "BARL anti-reflection multilayer",
        "status": "modelled",
        "config": "barl.layers",
    },
    {
        "feature": "FDTI / BDTI deep trench isolation",
        "status": "modelled",
        "config": "silicon.dti.mode/depth/width",
    },
    {
        "feature": "DTI conformal high-k passivation liner",
        "status": "modelled (new)",
        "config": "silicon.dti.liner",
    },
    {
        "feature": "Tapered DTI sidewall (etch profile)",
        "status": "modelled (new)",
        "config": "silicon.dti.taper_angle",
    },
    {
        "feature": "Backside inverted-pyramid light-trapping texture",
        "status": "modelled (new)",
        "config": "silicon.surface_texture",
    },
    {
        "feature": "Photodiode collection window",
        "status": "modelled",
        "config": "silicon.photodiode",
    },
    {
        "feature": "Composite / air-gap metal grid liner",
        "status": "roadmap",
        "config": "-",
    },
    {
        "feature": "In-pixel light pipe / inner lens",
        "status": "roadmap",
        "config": "-",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pixel structure realism report.")
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--date", default=date.today().isoformat())
    return parser.parse_args()


def eps_xz_section(
    stack: PixelStack,
    wavelength: float,
    nx: int = 220,
    nz: int = 360,
    n_lens_slices: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample Re(eps)(x, z) along the y-centre of the unit cell.

    Returns (x, z, eps_real) where eps_real has shape (nz, nx). This is exactly
    the permittivity the RCWA solver integrates over, so the figure faithfully
    shows the tapered/lined DTI, the graded backside texture, the color-filter
    relief, and the microlens residual base.
    """
    lx, ly = stack.domain_size
    z_min, z_max = stack.z_range
    x = np.linspace(0, lx, nx, endpoint=False)
    z = np.linspace(z_min, z_max, nz)

    slices = stack.get_layer_slices(wavelength, nx=nx, ny=max(nx, 8), n_lens_slices=n_lens_slices)
    # Sample through the centre of the first pixel (y = 0.5*pitch) rather than
    # the unit-cell centre, which would land on a pixel boundary and slice along
    # the horizontal DTI line.
    ny = slices[0].eps_grid.shape[0]
    y_center = 0.5 * stack.pitch
    y_row = int(np.clip(round(y_center / ly * ny), 0, ny - 1))
    eps_real = np.ones((nz, nx))
    for s in slices:
        zmask = (z >= s.z_start) & (z < s.z_end)
        if not np.any(zmask):
            continue
        row = np.real(s.eps_grid[y_row, :])
        eps_real[zmask, :] = row[np.newaxis, :]
    return x, z, eps_real


def plot_baseline_vs_realism(outpath: Path) -> dict[str, Any]:
    """Side-by-side eps(x,z) of a baseline vs an NIR realism-enhanced pixel."""
    baseline = PixelStack({"pixel": derive_parameters("generic_bsi", pitch=1.12)})
    nir = PixelStack({"pixel": derive_parameters("sample_p1p12um_nir")})
    wl = 0.70

    sections = [
        ("Baseline 1.12 um BSI", baseline),
        ("NIR-enhanced (IPA + lined DTI)", nir),
    ]
    computed = {label: eps_xz_section(stack, wl) for label, stack in sections}
    vmax = max(float(eps.max()) for _, _, eps in computed.values())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), constrained_layout=True)
    info: dict[str, Any] = {}
    for ax, (label, stack) in zip(axes, sections):
        x, z, eps = computed[label]
        im = ax.pcolormesh(x, z, eps, shading="auto", cmap="viridis", vmin=1.0, vmax=vmax)
        ax.set_title(f"{label}\nRe(eps) at y-centre, wl = {wl:.2f} um")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        ax.set_aspect("auto")
        fig.colorbar(im, ax=ax, label="Re(epsilon)")
        info[label] = {
            "total_height_um": float(stack.total_height),
            "n_silicon_slices": int(
                sum(
                    1
                    for s in stack.get_layer_slices(wl, nx=96, ny=96)
                    if s.name.startswith("silicon")
                )
            ),
        }
    fig.suptitle("Pixel structure realism: baseline vs NIR-enhanced (solver input)", fontsize=15)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)
    return info


def plot_silicon_zoom(outpath: Path) -> dict[str, Any]:
    """Zoom on the silicon backside: graded texture + tapered, lined DTI."""
    nir = PixelStack({"pixel": derive_parameters("sample_p1p12um_nir")})
    wl = 0.85
    x, z, eps = eps_xz_section(nir, wl, nx=300, nz=420)

    # Zoom to the top ~0.8 um of silicon where texture + DTI taper live.
    si_layer = next(layer for layer in nir.layers if layer.name == "silicon")
    z0 = si_layer.z_end - 0.8
    z1 = si_layer.z_end + 0.05
    zmask = (z >= z0) & (z <= z1)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 6.0), constrained_layout=True)
    im = ax.pcolormesh(x, z[zmask], eps[zmask, :], shading="auto", cmap="cividis")
    ax.set_title(
        "Silicon backside detail @ 0.85 um\n"
        "inverted-pyramid texture (graded Re(eps)) + tapered DTI with high-k liner"
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    fig.colorbar(im, ax=ax, label="Re(epsilon)")
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

    # Effective-index profile of the texture region (area-averaged sqrt(Re eps)).
    tex = nir._layer_configs["silicon"]["surface_texture"]
    tex_h = float(tex["height"])
    z_top = si_layer.z_end
    depth = np.linspace(0.0, tex_h, 30)
    neff = []
    for d in depth:
        col_z = z_top - d
        idx = int(np.argmin(np.abs(z - col_z)))
        neff.append(float(np.mean(np.sqrt(np.clip(eps[idx, :], 0, None)))))
    return {
        "texture_height_um": tex_h,
        "n_eff_surface": neff[0],
        "n_eff_apex": neff[-1],
    }


def main() -> None:
    args = parse_args()
    docs_root = args.docs.resolve()
    public_dir = docs_root / "public" / "reports" / "geometry" / "structure-realism"
    public_dir.mkdir(parents=True, exist_ok=True)

    compare_info = plot_baseline_vs_realism(public_dir / "baseline_vs_realism.png")
    zoom_info = plot_silicon_zoom(public_dir / "silicon_backside_zoom.png")

    metrics = {
        "generated_on": args.date,
        "compare": compare_info,
        "silicon_zoom": zoom_info,
        "coverage": COVERAGE,
    }
    (public_dir / "structure_realism_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    write_reports(docs_root, args.date, metrics)
    print(f"Wrote structure realism report and assets to {public_dir}")


def _coverage_table(lang: str) -> str:
    if lang == "ko":
        headers = ["실제 픽셀 요소", "COMPASS 상태", "config 키"]
    else:
        headers = ["Real-pixel feature", "COMPASS status", "config key"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * 3) + " |",
    ]
    for row in COVERAGE:
        lines.append(f"| {row['feature']} | {row['status']} | `{row['config']}` |")
    return "\n".join(lines)


def write_reports(docs_root: Path, generated_on: str, metrics: dict[str, Any]) -> None:
    reports = docs_root / "reports"
    reports_ko = docs_root / "ko" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    reports_ko.mkdir(parents=True, exist_ok=True)

    zoom = metrics["silicon_zoom"]
    n_surf = zoom["n_eff_surface"]
    n_apex = zoom["n_eff_apex"]

    en = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# Pixel Structure Realism Report",
            "",
            f"_Generated on {generated_on} from `compass.geometry.sample_pixels` and `PixelStack`._",
            "",
            "This report documents the structural-realism features COMPASS models for "
            "modern backside-illuminated (BSI) CMOS image-sensor pixels, and visualises "
            "them directly from the generated solver input. The cross-sections sample the "
            "real permittivity `Re(eps)(x, z)` the RCWA solver integrates, so this is "
            "geometry evidence rather than an optical-performance claim.",
            "",
            "## Why these features matter",
            "",
            "- **Backside inverted-pyramid texture (IPA).** A graded silicon fill fraction "
            "at the backside acts as a moth-eye anti-reflection / light-trapping layer. "
            "Published simulations and process reports show large near-infrared QE gains "
            "(up to ~3x at 850 nm and ~5x at 940 nm) when an IPA is combined with deep DTI.",
            "- **DTI conformal high-k liner.** Real backside DTI trenches are lined with a "
            "thin high-k film (Al2O3 / HfO2 / Ta2O5, ~30-100 nm) that passivates the etched "
            "silicon and carries a negative fixed charge. Optically it is a thin high-index "
            "ring between silicon and the lower-index trench fill.",
            "- **Tapered DTI sidewall.** Etched trenches narrow with depth; a vertical-wall "
            "idealisation over-counts the isolation oxide deep in the substrate.",
            "- **Microlens residual base.** Reflow / etch-back leaves a flat polymer slab "
            "under the curved cap; the lens is never zero-thickness at its edges.",
            "",
            "## Baseline vs NIR-enhanced pixel",
            "",
            "![Baseline vs realism-enhanced pixel](/reports/geometry/structure-realism/baseline_vs_realism.png)",
            "",
            "## Silicon backside detail",
            "",
            "The graded `Re(eps)` of the inverted-pyramid texture produces a smooth "
            "effective-index transition from the trench fill toward bulk silicon, while the "
            "tapered DTI trench and its high-k liner are visible at the pixel boundary.",
            "",
            "![Silicon backside detail](/reports/geometry/structure-realism/silicon_backside_zoom.png)",
            "",
            f"- Texture height: **{zoom['texture_height_um']:.3f} um**",
            f"- Area-averaged effective index across the texture: "
            f"**{n_surf:.2f}** at the surface to **{n_apex:.2f}** toward the apex "
            "(monotonic graded-index anti-reflection).",
            "",
            "## Capability coverage matrix",
            "",
            _coverage_table("en"),
            "",
            "## Reproduce this pixel",
            "",
            "```bash",
            "python scripts/run_simulation.py pixel=sample_p1p12um_nir",
            "```",
            "",
            "Or derive a config in Python:",
            "",
            "```python",
            "from compass.geometry.sample_pixels import derive_parameters",
            "from compass.geometry.pixel_stack import PixelStack",
            "",
            'cfg = derive_parameters("sample_p1p12um_nir")',
            'stack = PixelStack({"pixel": cfg})',
            "```",
            "",
            "## Regeneration",
            "",
            "```bash",
            "python scripts/generate_realism_report.py",
            "```",
            "",
            "Generated metrics are stored at "
            "`docs/public/reports/geometry/structure-realism/structure_realism_metrics.json`.",
            "",
        ]
    )

    ko = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# 픽셀 구조 현실성 리포트",
            "",
            f"_생성일: {generated_on}. `compass.geometry.sample_pixels`와 `PixelStack`에서 생성._",
            "",
            "이 리포트는 최신 후면 조사형(BSI) CMOS 이미지 센서 픽셀에 대해 COMPASS가 "
            "모델링하는 구조적 현실성 요소를 문서화하고, 생성된 solver 입력에서 직접 "
            "시각화한다. 단면은 RCWA solver가 적분하는 실제 유전율 `Re(eps)(x, z)`를 "
            "샘플링하므로, 광학 성능 주장이 아니라 geometry evidence다.",
            "",
            "## 왜 중요한가",
            "",
            "- **후면 역피라미드 텍스처(IPA).** 후면의 그라데이션 실리콘 충전율은 moth-eye "
            "반사방지/광 트래핑 층으로 작동한다. 공개된 시뮬레이션·공정 보고서는 IPA를 "
            "깊은 DTI와 결합할 때 근적외선 QE가 크게(850 nm에서 최대 ~3배, 940 nm에서 "
            "~5배) 증가함을 보인다.",
            "- **DTI 컨포멀 high-k 라이너.** 실제 후면 DTI 트렌치는 식각된 실리콘을 "
            "패시베이션하고 음의 고정 전하를 갖는 얇은 high-k 막(Al2O3 / HfO2 / Ta2O5, "
            "약 30-100 nm)으로 라이닝된다. 광학적으로는 실리콘과 저굴절 충전재 사이의 "
            "얇은 고굴절 링이다.",
            "- **테이퍼 DTI 측벽.** 식각된 트렌치는 깊이에 따라 좁아진다. 수직 측벽 "
            "이상화는 기판 깊은 곳의 격리 산화막을 과대 계산한다.",
            "- **마이크로렌즈 잔류층.** Reflow / etch-back은 곡면 캡 아래에 평탄한 폴리머 "
            "슬랩을 남긴다. 렌즈는 가장자리에서 두께가 0이 아니다.",
            "",
            "## 기준 픽셀 vs NIR 강화 픽셀",
            "",
            "![Baseline vs realism-enhanced pixel](/reports/geometry/structure-realism/baseline_vs_realism.png)",
            "",
            "## 실리콘 후면 상세",
            "",
            "역피라미드 텍스처의 그라데이션 `Re(eps)`는 트렌치 충전재에서 벌크 실리콘으로 "
            "매끄러운 유효 굴절률 전이를 만들고, 픽셀 경계에서는 테이퍼 DTI 트렌치와 "
            "high-k 라이너가 보인다.",
            "",
            "![Silicon backside detail](/reports/geometry/structure-realism/silicon_backside_zoom.png)",
            "",
            f"- 텍스처 높이: **{zoom['texture_height_um']:.3f} um**",
            f"- 텍스처 전체의 면적 평균 유효 굴절률: 표면 **{n_surf:.2f}** → 정점 방향 "
            f"**{n_apex:.2f}** (단조 증가하는 graded-index 반사방지).",
            "",
            "## 기능 커버리지 매트릭스",
            "",
            _coverage_table("ko"),
            "",
            "## 이 픽셀 재현",
            "",
            "```bash",
            "python scripts/run_simulation.py pixel=sample_p1p12um_nir",
            "```",
            "",
            "또는 Python에서 config를 유도한다:",
            "",
            "```python",
            "from compass.geometry.sample_pixels import derive_parameters",
            "from compass.geometry.pixel_stack import PixelStack",
            "",
            'cfg = derive_parameters("sample_p1p12um_nir")',
            'stack = PixelStack({"pixel": cfg})',
            "```",
            "",
            "## 재생성",
            "",
            "```bash",
            "python scripts/generate_realism_report.py",
            "```",
            "",
            "생성 metric은 "
            "`docs/public/reports/geometry/structure-realism/structure_realism_metrics.json`에 저장된다.",
            "",
        ]
    )

    (reports / "pixel-structure-realism.md").write_text(en, encoding="utf-8")
    (reports_ko / "pixel-structure-realism.md").write_text(ko, encoding="utf-8")


if __name__ == "__main__":
    main()
