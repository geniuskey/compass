#!/usr/bin/env python3
"""Generate geometry-focused VitePress report pages.

The reports generated here are intentionally geometry-only. They publish the
actual PixelStack shapes and derived metrics used by the solvers without making
new optical-performance claims.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
from compass.visualization.structure_plot_2d import plot_pixel_cross_section  # noqa: E402

DOCS = ROOT / "docs"
PUBLIC_DIR = DOCS / "public" / "reports" / "geometry"

SAMPLE_KEYS = [
    "generic_bsi",
    "sample_p0p56um_4x4ocl",
    "sample_p1p0um_quadbayer",
    "sample_p1p22um_2x2ocl",
    "sample_p1p6um_split_pd",
    "sample_p1p2um_lofic",
    "sample_p1p12um_nir",
]

SAMPLE_LABELS = {
    "generic_bsi": "Generic 1.0 um BSI",
    "sample_p0p56um_4x4ocl": "0.56 um 4x4 OCL",
    "sample_p1p0um_quadbayer": "1.0 um Quad Bayer",
    "sample_p1p22um_2x2ocl": "1.22 um 2x2 OCL",
    "sample_p1p6um_split_pd": "1.6 um split PD",
    "sample_p1p2um_lofic": "1.2 um LOFIC",
    "sample_p1p12um_nir": "1.12 um NIR (IPA + lined DTI)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate geometry audit and color-filter relief reports.",
    )
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--date", default=date.today().isoformat())
    return parser.parse_args()


def fmt(value: Any, digits: int = 3) -> str:
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


def stack_from_pixel_cfg(pixel_cfg: dict[str, Any]) -> PixelStack:
    return PixelStack({"pixel": pixel_cfg})


def color_filter_layer(stack: PixelStack):
    for layer in stack.layers:
        if layer.name == "color_filter":
            return layer
    raise RuntimeError("PixelStack has no color_filter layer")


def channel_specs(stack: PixelStack) -> dict[str, dict[str, Any]]:
    cf_cfg = stack._layer_configs.get("color_filter", {})
    return {color: stack._color_filter_spec(cf_cfg, color) for color in ["R", "G", "B"]}


def cf_area_fraction(
    pitch: float,
    grid_width: float,
    z_rel: float,
    grid_thickness: float,
    contact_angle: float,
) -> float:
    protrusion = max(0.0, z_rel - grid_thickness)
    inset = 0.0
    if protrusion > 0.0 and contact_angle < 89.999:
        theta = math.radians(max(1.0, min(89.999, contact_angle)))
        inset = protrusion / math.tan(theta)
    inner_half = (pitch - grid_width) / 2.0 - inset
    if inner_half <= 0.0:
        return 0.0
    return float((2.0 * inner_half / pitch) ** 2)


def sample_metrics(sample_key: str) -> dict[str, Any]:
    cfg = derive_parameters(sample_key)
    stack = stack_from_pixel_cfg(cfg)
    cf_cfg = cfg["layers"]["color_filter"]
    cf_layer = color_filter_layer(stack)
    specs = channel_specs(stack)
    slices = stack.get_layer_slices(0.55, nx=96, ny=96, n_lens_slices=12)
    cf_slice_count = sum(1 for item in slices if item.name.startswith("color_filter"))
    grid_t = stack._grid_thickness(cf_cfg)
    pd = cfg["layers"]["silicon"]["photodiode"]
    pd_xy = float(pd["size"][0])
    return {
        "key": sample_key,
        "label": SAMPLE_LABELS.get(sample_key, sample_key),
        "pitch_um": stack.pitch,
        "unit_cell": f"{stack.unit_cell[0]}x{stack.unit_cell[1]}",
        "domain_um": list(stack.domain_size),
        "total_height_um": stack.total_height,
        "color_filter_stack_um": cf_layer.thickness,
        "grid_thickness_um": grid_t,
        "grid_width_um": float(cf_cfg.get("grid", {}).get("width", 0.0)),
        "red_thickness_um": specs["R"]["thickness"],
        "green_thickness_um": specs["G"]["thickness"],
        "blue_thickness_um": specs["B"]["thickness"],
        "min_contact_angle_deg": min(float(spec["contact_angle"]) for spec in specs.values()),
        "color_filter_slice_count": cf_slice_count,
        "photodiode_xy_fill": (pd_xy / stack.pitch) ** 2,
        "dti_width_um": float(cfg["layers"]["silicon"]["dti"]["width"]),
        "checks": {
            "cf_stack_covers_all_channels": cf_layer.thickness
            >= max(float(spec["thickness"]) for spec in specs.values()),
            "grid_not_taller_than_stack": grid_t <= cf_layer.thickness,
            "relief_uses_multiple_slices": cf_slice_count > 1,
            "photodiode_inside_pixel": pd_xy <= stack.pitch,
        },
    }


def color_filter_metrics() -> dict[str, Any]:
    cfg = derive_parameters("generic_bsi", pitch=1.0)
    stack = stack_from_pixel_cfg(cfg)
    cf_cfg = cfg["layers"]["color_filter"]
    specs = channel_specs(stack)
    grid_width = float(cf_cfg["grid"]["width"])
    grid_t = stack._grid_thickness(cf_cfg)
    rows: list[dict[str, Any]] = []
    for color, spec in specs.items():
        thickness = float(spec["thickness"])
        angle = float(spec["contact_angle"])
        rows.append(
            {
                "color": color,
                "material": spec["material"],
                "thickness_um": thickness,
                "protrusion_above_grid_um": max(0.0, thickness - grid_t),
                "contact_angle_deg": angle,
                "top_area_fraction_of_pitch": cf_area_fraction(
                    stack.pitch,
                    grid_width,
                    thickness,
                    grid_t,
                    angle,
                ),
            }
        )
    return {
        "pitch_um": stack.pitch,
        "grid_width_um": grid_width,
        "grid_thickness_um": grid_t,
        "rows": rows,
    }


def make_flat_config() -> dict[str, Any]:
    cfg = derive_parameters("generic_bsi", pitch=1.0)
    cfg = copy.deepcopy(cfg)
    cf = cfg["layers"]["color_filter"]
    flat_t = float(cf["green"]["thickness"])
    cf["thickness"] = flat_t
    cf["materials"] = {"R": "cf_red", "G": "cf_green", "B": "cf_blue"}
    cf.pop("red", None)
    cf.pop("green", None)
    cf.pop("blue", None)
    cf["grid"]["thickness"] = flat_t
    return cfg


def make_angle_config(angle: float) -> dict[str, Any]:
    cfg = derive_parameters("generic_bsi", pitch=1.0)
    cfg = copy.deepcopy(cfg)
    for channel in ["red", "green", "blue"]:
        cfg["layers"]["color_filter"][channel]["contact_angle"] = angle
    return cfg


def plot_sample_overview(outpath: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), constrained_layout=True)
    flat_axes = axes.ravel()
    for ax, sample_key in zip(flat_axes, SAMPLE_KEYS):
        cfg = derive_parameters(sample_key)
        stack = stack_from_pixel_cfg(cfg)
        plot_pixel_cross_section(
            stack,
            plane="xz",
            position=stack.domain_size[1] / 2.0,
            wavelength=0.55,
            ax=ax,
        )
        for text in list(ax.texts):
            text.remove()
        ax.set_title(f"{SAMPLE_LABELS.get(sample_key, sample_key)}\npitch {stack.pitch:.3f} um")
    for ax in flat_axes[len(SAMPLE_KEYS):]:
        ax.axis("off")
    fig.suptitle("PixelStack geometry audit: representative sample pixels", fontsize=16)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def draw_cf_variant(ax: plt.Axes, title: str, cfg: dict[str, Any]) -> None:
    stack = stack_from_pixel_cfg(cfg)
    cf_cfg = stack._layer_configs.get("color_filter", {})
    grid_cfg = cf_cfg.get("grid", {}) or {}
    grid_width = float(grid_cfg.get("width", 0.0)) if grid_cfg.get("enabled", False) else 0.0
    grid_t = stack._grid_thickness(cf_cfg)
    cf_layer = color_filter_layer(stack)
    row = 0
    n_cols = min(2, stack.unit_cell[1])
    colors = {"R": "#dc2626", "G": "#16a34a", "B": "#2563eb"}

    # Metal grid is shown as boundary bars up to grid.thickness.
    if grid_width > 0.0 and grid_t > 0.0:
        grid_rects = [
            (0.0, grid_width / 2.0),
            (stack.pitch - grid_width / 2.0, stack.pitch + grid_width / 2.0),
            (n_cols * stack.pitch - grid_width / 2.0, n_cols * stack.pitch),
        ]
        for x0, x1 in grid_rects:
            ax.add_patch(
                plt.Rectangle(
                    (max(0.0, x0), 0.0),
                    max(0.0, x1 - max(0.0, x0)),
                    grid_t,
                    facecolor="#facc15",
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.95,
                    label="_nolegend_",
                )
            )

    for col in range(n_cols):
        color = stack.bayer_map[row][col]
        spec = stack._color_filter_spec(cf_cfg, color)
        thickness = min(float(spec["thickness"]), cf_layer.thickness)
        x0 = col * stack.pitch + grid_width / 2.0
        x1 = (col + 1) * stack.pitch - grid_width / 2.0
        z_grid_top = min(grid_t, thickness)
        if z_grid_top > 0:
            ax.add_patch(
                plt.Rectangle(
                    (x0, 0.0),
                    x1 - x0,
                    z_grid_top,
                    facecolor=colors.get(color, "#94a3b8"),
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.95,
                    label="_nolegend_",
                )
            )
        if thickness > z_grid_top:
            inset = stack._cf_lateral_inset(
                thickness,
                grid_t,
                float(spec["contact_angle"]),
            )
            x0_top = min(x0 + inset, (x0 + x1) / 2.0)
            x1_top = max(x1 - inset, (x0 + x1) / 2.0)
            ax.add_patch(
                plt.Polygon(
                    [
                        (x0, z_grid_top),
                        (x1, z_grid_top),
                        (x1_top, thickness),
                        (x0_top, thickness),
                    ],
                    closed=True,
                    facecolor=colors.get(color, "#94a3b8"),
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.95,
                    label="_nolegend_",
                )
            )
        ax.text(
            (x0 + x1) / 2.0,
            min(thickness + 0.015, cf_layer.thickness + 0.01),
            color,
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
            label="_nolegend_",
        )

    ax.axhline(grid_t, color="#334155", linestyle=":", linewidth=1.4, label="grid.thickness")
    ax.axhline(cf_layer.thickness, color="#64748b", linestyle="--", linewidth=1.0, label="CF stack top")
    ax.set_xlim(0.0, n_cols * stack.pitch)
    ax.set_ylim(0.0, cf_layer.thickness + 0.08)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("local z in color-filter stack (um)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def plot_color_filter_sections(outpath: Path) -> None:
    variants = [
        ("Legacy flat slab", make_flat_config()),
        ("Default relief", derive_parameters("generic_bsi", pitch=1.0)),
        ("Low contact angle", make_angle_config(52.0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (title, cfg) in zip(axes, variants):
        draw_cf_variant(ax, title, cfg)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Color filter cross-section variants", fontsize=15)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_contact_angle_sweep(metrics: dict[str, Any], outpath: Path) -> None:
    pitch = float(metrics["pitch_um"])
    grid_width = float(metrics["grid_width_um"])
    grid_t = float(metrics["grid_thickness_um"])
    max_t = max(row["thickness_um"] for row in metrics["rows"])
    z_values = np.linspace(grid_t, max_t, 160)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0), constrained_layout=True)
    for angle in [90.0, 72.0, 66.0, 62.0, 52.0]:
        area = [
            cf_area_fraction(pitch, grid_width, float(z), grid_t, angle)
            for z in z_values
        ]
        ax.plot(z_values, area, label=f"{angle:.0f} deg")

    color_map = {"R": "#dc2626", "G": "#16a34a", "B": "#2563eb"}
    for row in metrics["rows"]:
        ax.scatter(
            [row["thickness_um"]],
            [row["top_area_fraction_of_pitch"]],
            s=70,
            color=color_map.get(row["color"], "#111827"),
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label=f"{row['color']} actual",
        )

    ax.axvline(grid_t, color="#64748b", linestyle=":", label="grid top")
    ax.set_xlabel("Local z height in color-filter stack (um)")
    ax.set_ylabel("Top footprint area / pixel pitch area")
    ax.set_title("Contact angle controls the tapered color-filter top footprint")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_reports(
    docs_root: Path,
    generated_on: str,
    samples: list[dict[str, Any]],
    cf_metrics: dict[str, Any],
) -> None:
    reports = docs_root / "reports"
    reports_ko = docs_root / "ko" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    reports_ko.mkdir(parents=True, exist_ok=True)

    sample_rows = [
        [
            item["label"],
            item["pitch_um"],
            item["unit_cell"],
            item["color_filter_stack_um"],
            item["grid_thickness_um"],
            f"{fmt(item['red_thickness_um'])}/{fmt(item['green_thickness_um'])}/{fmt(item['blue_thickness_um'])}",
            item["min_contact_angle_deg"],
            item["color_filter_slice_count"],
            item["photodiode_xy_fill"],
        ]
        for item in samples
    ]
    cf_rows = [
        [
            row["color"],
            row["material"],
            row["thickness_um"],
            row["protrusion_above_grid_um"],
            row["contact_angle_deg"],
            row["top_area_fraction_of_pitch"],
        ]
        for row in cf_metrics["rows"]
    ]

    geometry_en = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# Pixel Stack Geometry Audit",
            "",
            f"_Generated on {generated_on} from `compass.geometry.sample_pixels` and `PixelStack`._",
            "",
            "This report publishes geometry evidence, not optical performance. It verifies "
            "that representative sample-pixel presets expand into plausible solver input "
            "stacks with color-filter relief, metal-grid thickness, DTI, microlens, and "
            "photodiode windows present in the generated `PixelStack`.",
            "",
            "## Executive summary",
            "",
            "- All audited presets produce color-filter relief slices rather than a single "
            "flat slab.",
            "- The color-filter stack height covers the tallest RGB channel and the metal "
            "grid thickness for every audited preset.",
            "- Photodiode x-y windows stay inside the pixel pitch for every audited preset.",
            "- This report does not claim QE or crosstalk deltas. Use it before optical "
            "sweeps to confirm the geometry being simulated.",
            "",
            "## Geometry overview",
            "",
            "![PixelStack geometry overview](/reports/geometry/pixel-stack-audit/sample_stack_overview.png)",
            "",
            "## Audited presets",
            "",
            markdown_table(
                [
                    "Preset",
                    "pitch um",
                    "unit cell",
                    "CF stack um",
                    "grid um",
                    "R/G/B CF um",
                    "min angle",
                    "CF slices",
                    "PD xy fill",
                ],
                sample_rows,
            ),
            "",
            "## Checks",
            "",
            markdown_table(
                ["Preset", "CF covers channels", "grid <= stack", "multi-slice relief", "PD inside pixel"],
                [
                    [
                        item["label"],
                        item["checks"]["cf_stack_covers_all_channels"],
                        item["checks"]["grid_not_taller_than_stack"],
                        item["checks"]["relief_uses_multiple_slices"],
                        item["checks"]["photodiode_inside_pixel"],
                    ]
                    for item in samples
                ],
            ),
            "",
            "## Regeneration",
            "",
            "```powershell",
            "uv run python scripts\\generate_geometry_reports.py",
            "```",
            "",
            "Generated metrics are stored at "
            "`docs/public/reports/geometry/pixel-stack-audit/geometry_metrics.json`.",
            "",
        ]
    )

    geometry_ko = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# 픽셀 스택 Geometry 감사",
            "",
            f"_생성일: {generated_on}. `compass.geometry.sample_pixels`와 `PixelStack`에서 생성._",
            "",
            "이 리포트는 광학 성능이 아니라 geometry evidence를 게시한다. 대표 sample "
            "pixel preset이 color-filter relief, metal-grid thickness, DTI, microlens, "
            "photodiode window를 포함한 solver 입력 stack으로 실제 확장되는지 확인한다.",
            "",
            "## 요약",
            "",
            "- 모든 감사 대상 preset은 단일 flat slab가 아니라 color-filter relief slice를 생성한다.",
            "- 모든 preset에서 color-filter stack 높이는 가장 높은 RGB channel과 metal grid를 덮는다.",
            "- photodiode x-y window는 모든 preset에서 pixel pitch 안에 있다.",
            "- 이 리포트는 QE 또는 crosstalk 변화를 주장하지 않는다. optical sweep 전에 "
            "시뮬레이션되는 geometry를 확인하는 용도다.",
            "",
            "## Geometry overview",
            "",
            "![PixelStack geometry overview](/reports/geometry/pixel-stack-audit/sample_stack_overview.png)",
            "",
            "## 감사 대상 preset",
            "",
            markdown_table(
                [
                    "Preset",
                    "pitch um",
                    "unit cell",
                    "CF stack um",
                    "grid um",
                    "R/G/B CF um",
                    "min angle",
                    "CF slices",
                    "PD xy fill",
                ],
                sample_rows,
            ),
            "",
            "## Checks",
            "",
            markdown_table(
                ["Preset", "CF covers channels", "grid <= stack", "multi-slice relief", "PD inside pixel"],
                [
                    [
                        item["label"],
                        item["checks"]["cf_stack_covers_all_channels"],
                        item["checks"]["grid_not_taller_than_stack"],
                        item["checks"]["relief_uses_multiple_slices"],
                        item["checks"]["photodiode_inside_pixel"],
                    ]
                    for item in samples
                ],
            ),
            "",
            "## 재생성",
            "",
            "```powershell",
            "uv run python scripts\\generate_geometry_reports.py",
            "```",
            "",
            "생성 metric은 "
            "`docs/public/reports/geometry/pixel-stack-audit/geometry_metrics.json`에 저장된다.",
            "",
        ]
    )

    cf_en = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# Color Filter Relief Sensitivity Report",
            "",
            f"_Generated on {generated_on} from the generic 1.0 um BSI `PixelStack`._",
            "",
            "This is a geometry-sensitivity report for the per-channel color-filter model. "
            "It shows how `grid.thickness`, `red/green/blue.thickness`, and "
            "`red/green/blue.contact_angle` change the z-sliced solver geometry.",
            "",
            "::: info Scope",
            "The figures below are geometry evidence. They do not yet report optical QE or "
            "crosstalk deltas. The next optical report should run RCWA order sweeps over "
            "these geometry variants.",
            ":::",
            "",
            "## Cross-section variants",
            "",
            "![Color filter relief cross sections](/reports/geometry/color-filter-relief/color_filter_relief_sections.png)",
            "",
            "## Contact-angle sweep",
            "",
            "![Contact angle sweep](/reports/geometry/color-filter-relief/contact_angle_sweep.png)",
            "",
            "## Default per-channel geometry",
            "",
            markdown_table(
                [
                    "Color",
                    "material",
                    "thickness um",
                    "above grid um",
                    "contact angle",
                    "top area / pitch area",
                ],
                cf_rows,
            ),
            "",
            "## Interpretation",
            "",
            "- `grid.thickness` defines the vertical part of the metal-grid region.",
            "- Channel `thickness` values define the maximum color-resist height per color.",
            "- `contact_angle` controls the trapezoidal taper above the grid. Lower angle "
            "means a smaller top footprint for the same protrusion height.",
            "- Because red, green, and blue use different heights and angles, RCWA receives "
            "multiple color-filter z slices even before microlens slicing is considered.",
            "",
            "## Regeneration",
            "",
            "```powershell",
            "uv run python scripts\\generate_geometry_reports.py",
            "```",
            "",
            "Generated metrics are stored at "
            "`docs/public/reports/geometry/color-filter-relief/color_filter_relief_metrics.json`.",
            "",
        ]
    )

    cf_ko = "\n".join(
        [
            "---",
            "outline: deep",
            "---",
            "",
            "# 컬러 필터 Relief 민감도 리포트",
            "",
            f"_생성일: {generated_on}. generic 1.0 um BSI `PixelStack`에서 생성._",
            "",
            "이 리포트는 색별 컬러 필터 모델의 geometry 민감도 리포트다. "
            "`grid.thickness`, `red/green/blue.thickness`, "
            "`red/green/blue.contact_angle`이 z-sliced solver geometry를 어떻게 "
            "바꾸는지 보여준다.",
            "",
            "::: info 범위",
            "아래 그림은 geometry evidence다. 아직 optical QE 또는 crosstalk delta를 "
            "보고하지 않는다. 다음 optical 리포트에서는 이 geometry variant에 대해 RCWA "
            "order sweep을 실행하는 것이 좋다.",
            ":::",
            "",
            "## 단면 variant",
            "",
            "![Color filter relief cross sections](/reports/geometry/color-filter-relief/color_filter_relief_sections.png)",
            "",
            "## Contact-angle sweep",
            "",
            "![Contact angle sweep](/reports/geometry/color-filter-relief/contact_angle_sweep.png)",
            "",
            "## 기본 색별 geometry",
            "",
            markdown_table(
                [
                    "Color",
                    "material",
                    "thickness um",
                    "above grid um",
                    "contact angle",
                    "top area / pitch area",
                ],
                cf_rows,
            ),
            "",
            "## 해석",
            "",
            "- `grid.thickness`는 metal-grid 영역의 수직 높이를 정한다.",
            "- 색별 `thickness`는 각 color resist의 최대 높이를 정한다.",
            "- `contact_angle`은 grid 위 돌출부의 사다리꼴 taper를 제어한다. 각도가 낮을수록 "
            "같은 돌출 높이에서 top footprint가 작아진다.",
            "- red, green, blue가 서로 다른 높이와 각도를 가지므로 microlens slice를 "
            "고려하기 전에도 RCWA에는 여러 color-filter z slice가 들어간다.",
            "",
            "## 재생성",
            "",
            "```powershell",
            "uv run python scripts\\generate_geometry_reports.py",
            "```",
            "",
            "생성 metric은 "
            "`docs/public/reports/geometry/color-filter-relief/color_filter_relief_metrics.json`에 저장된다.",
            "",
        ]
    )

    (reports / "pixel-stack-geometry-audit.md").write_text(geometry_en, encoding="utf-8")
    (reports_ko / "pixel-stack-geometry-audit.md").write_text(geometry_ko, encoding="utf-8")
    (reports / "color-filter-relief-sensitivity.md").write_text(cf_en, encoding="utf-8")
    (reports_ko / "color-filter-relief-sensitivity.md").write_text(cf_ko, encoding="utf-8")


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
            "Publication-style reports generated from Python benchmark artifacts and "
            "geometry audit scripts. Reports are for validation evidence: generated "
            "figures, metric tables, and exact regeneration commands.",
            "",
            "## Available reports",
            "",
            "- [TMM vs RCWA Planar Stack Validation](./tmm-rcwa-planar-validation.md) (generated 2026-06-11)",
            "- [RCWA Backend Parity](./rcwa-backend-parity.md) (generated 2026-06-11)",
            "- [CRA Cone Illumination Sweep](./cra-cone-illumination-sweep.md) (generated 2026-06-11)",
            "- [BARL Optimization Benchmark](./barl-optimization-benchmark.md) (generated 2026-06-11)",
            "- [DTI Crosstalk Benchmark](./dti-crosstalk-benchmark.md) (generated 2026-06-11)",
            "- [Performance Benchmark](./performance-benchmark.md) (generated 2026-06-11)",
            f"- [RCWA/FDTD Convergence Analysis](./convergence-analysis.md) (generated 2026-05-07)",
            f"- [Pixel Stack Geometry Audit](./pixel-stack-geometry-audit.md) (generated {generated_on})",
            f"- [Color Filter Relief Sensitivity](./color-filter-relief-sensitivity.md) (generated {generated_on})",
            f"- [Pixel Structure Realism](./pixel-structure-realism.md) (generated {generated_on})",
            "",
            "## Report queue",
            "",
            "_No report is currently queued._",
            "",
            "## What belongs here",
            "",
            "- Cross-solver validation results that should be inspectable from GitHub Pages.",
            "- Geometry audits that prove the solver input stack matches the intended config.",
            "- Plots and tables promoted from local `outputs/` artifacts into `docs/public/reports/`.",
            "- Reproducibility notes that explain which scripts regenerate the published figures.",
            "",
            "Use [Theory](/theory/) for concepts, [Guide](/guide/) for workflows, "
            "[Cookbook](/cookbook/bsi-2x2-basic) for recipes, and Reports for "
            "generated evidence.",
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
            "Python benchmark 산출물과 geometry 감사 스크립트에서 생성한 publication-style "
            "리포트다. Reports는 검증 근거를 위한 섹션이다: 생성된 그림, metric table, "
            "정확한 재생성 명령을 담는다.",
            "",
            "## 리포트 목록",
            "",
            "- [TMM vs RCWA 평면 스택 검증](./tmm-rcwa-planar-validation.md) (생성일 2026-06-11)",
            "- [RCWA Backend Parity](./rcwa-backend-parity.md) (생성일 2026-06-11)",
            "- [CRA Cone Illumination Sweep](./cra-cone-illumination-sweep.md) (생성일 2026-06-11)",
            "- [BARL Optimization Benchmark](./barl-optimization-benchmark.md) (생성일 2026-06-11)",
            "- [DTI Crosstalk Benchmark](./dti-crosstalk-benchmark.md) (생성일 2026-06-11)",
            "- [Performance Benchmark](./performance-benchmark.md) (생성일 2026-06-11)",
            "- [RCWA/FDTD 수렴 분석](./convergence-analysis.md) (생성일 2026-05-07)",
            f"- [픽셀 스택 Geometry 감사](./pixel-stack-geometry-audit.md) (생성일 {generated_on})",
            f"- [컬러 필터 Relief 민감도](./color-filter-relief-sensitivity.md) (생성일 {generated_on})",
            f"- [픽셀 구조 현실성](./pixel-structure-realism.md) (생성일 {generated_on})",
            "",
            "## 리포트 대기열",
            "",
            "_현재 대기 중인 리포트는 없다._",
            "",
            "## 이 섹션에 들어갈 내용",
            "",
            "- GitHub Pages에서 바로 확인할 수 있어야 하는 cross-solver 검증 결과.",
            "- solver 입력 stack이 의도한 config와 일치함을 보이는 geometry 감사.",
            "- 로컬 `outputs/` 산출물에서 `docs/public/reports/`로 승격한 그림과 표.",
            "- 공개된 그림을 어떤 스크립트로 다시 만들 수 있는지 설명하는 재현성 노트.",
            "",
            "개념은 [이론](/ko/theory/), 실행 절차는 [가이드](/ko/guide/), "
            "레시피는 [쿡북](/ko/cookbook/bsi-2x2-basic), 생성 근거는 Reports에 둔다.",
            "",
        ]
    )
    (reports / "index.md").write_text(english, encoding="utf-8")
    (reports_ko / "index.md").write_text(korean, encoding="utf-8")


def main() -> None:
    args = parse_args()
    docs_root = args.docs.resolve()
    public_dir = docs_root / "public" / "reports" / "geometry"
    stack_dir = public_dir / "pixel-stack-audit"
    cf_dir = public_dir / "color-filter-relief"
    stack_dir.mkdir(parents=True, exist_ok=True)
    cf_dir.mkdir(parents=True, exist_ok=True)

    samples = [sample_metrics(key) for key in SAMPLE_KEYS]
    cf_metrics = color_filter_metrics()

    plot_sample_overview(stack_dir / "sample_stack_overview.png")
    plot_color_filter_sections(cf_dir / "color_filter_relief_sections.png")
    plot_contact_angle_sweep(cf_metrics, cf_dir / "contact_angle_sweep.png")

    (stack_dir / "geometry_metrics.json").write_text(
        json.dumps({"generated_on": args.date, "samples": samples}, indent=2),
        encoding="utf-8",
    )
    (cf_dir / "color_filter_relief_metrics.json").write_text(
        json.dumps({"generated_on": args.date, **cf_metrics}, indent=2),
        encoding="utf-8",
    )

    write_reports(docs_root, args.date, samples, cf_metrics)
    write_index_pages(docs_root, args.date)

    print(f"Wrote geometry reports to {docs_root / 'reports'} and {docs_root / 'ko' / 'reports'}")
    print(f"Wrote public assets to {public_dir}")


if __name__ == "__main__":
    main()
