#!/usr/bin/env python3
"""Generate visual RCWA/FDTD-style optical simulation plots for a CMOS pixel.

This script is intentionally Python-level and non-interactive. It builds a
virtual CMOS image-sensor pixel stack, renders the solver input geometry, and
generates deterministic visual artifacts for two lightweight optical models:

* RCWA-style Fourier slice propagation through z-layer permittivity slices.
* FDTD-style scalar wave propagation through an XZ permittivity cross-section.

The models are visual test surrogates. They are designed to make geometry,
field confinement, photodiode collection, and crosstalk visible even when the
heavy optional RCWA/FDTD backend packages are not installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.colors import LogNorm
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.geometry.pixel_stack import PixelStack  # noqa: E402
from compass.materials.database import MaterialDB  # noqa: E402


@dataclass(frozen=True)
class XZGeometry:
    """XZ cross-section sampled from a PixelStack."""

    x: np.ndarray
    z: np.ndarray
    eps_xz: np.ndarray
    y_position: float
    layer_bounds: list[tuple[str, float, float]]
    photodiode_boxes: list[tuple[str, float, float, float, float]]


@dataclass
class RcwaVisualResult:
    """Visual result from the Fourier-slice RCWA-style propagation."""

    wavelengths: np.ndarray
    x: np.ndarray
    z: np.ndarray
    intensities: dict[float, np.ndarray]
    qe_per_pixel: dict[str, list[float]]
    crosstalk: dict[float, float]
    summary: dict[str, Any]


@dataclass
class FdtdVisualResult:
    """Visual result from scalar FDTD-style propagation."""

    wavelength: float
    x: np.ndarray
    z: np.ndarray
    snapshots: list[tuple[int, np.ndarray]]
    average_intensity: np.ndarray
    energy_trace: np.ndarray
    qe_per_pixel: dict[str, float]
    crosstalk: float
    summary: dict[str, Any]


@dataclass(frozen=True)
class SourceSettings:
    """Shared source/crosstalk definition for RCWA/FDTD visual alignment."""

    mode: str
    reference_pixel: tuple[int, int]
    active_columns: tuple[int, ...]
    reference_label: str
    y_position: float
    tilt_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visual CMOS pixel optical simulation artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pixel" / "default_bsi_1um.yaml",
        help="Pixel YAML config to visualize.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "visual_cmos_optical_simulation",
        help="Output directory for plots and report files.",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=180,
        help="Lateral grid resolution for visual simulations.",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=260,
        help="Vertical grid resolution for visual simulations.",
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=[0.45, 0.55, 0.65],
        help="Wavelengths in um for RCWA-style visual propagation.",
    )
    parser.add_argument(
        "--fdtd-wavelength",
        type=float,
        default=0.55,
        help="Wavelength in um for FDTD-style visual propagation.",
    )
    parser.add_argument(
        "--fdtd-steps",
        type=int,
        default=2200,
        help="Number of scalar FDTD time steps.",
    )
    parser.add_argument(
        "--source-mode",
        choices=["reference-pixel", "all-pixels"],
        default="reference-pixel",
        help="Illumination mode. Use reference-pixel for meaningful crosstalk comparison.",
    )
    parser.add_argument(
        "--reference-pixel",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=[0, 0],
        help="Reference pixel row/column used for single-pixel illumination and crosstalk.",
    )
    parser.add_argument(
        "--tilt-deg",
        type=float,
        default=0.0,
        help="Shared source tilt angle in degrees for both visual models.",
    )
    parser.add_argument(
        "--dti-mode",
        choices=["config", "fdti", "bdti"],
        default="config",
        help="DTI layout override. FDTI uses full-depth trench; BDTI uses --bdti-depth.",
    )
    parser.add_argument(
        "--bdti-depth",
        type=float,
        default=None,
        help="BDTI trench depth in um. Ignored unless --dti-mode bdti is selected.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use lower resolution for fast smoke checks.",
    )
    return parser.parse_args()


def load_pixel_stack(
    config_path: Path,
    dti_mode: str = "config",
    bdti_depth: float | None = None,
) -> PixelStack:
    raw = OmegaConf.load(config_path)
    config = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping config from {config_path}")
    apply_dti_override(config, dti_mode, bdti_depth)
    return PixelStack(config, MaterialDB())


def apply_dti_override(config: dict[str, Any], dti_mode: str, bdti_depth: float | None) -> None:
    pixel_cfg = config.setdefault("pixel", {})
    layers_cfg = pixel_cfg.setdefault("layers", {})
    silicon_cfg = layers_cfg.setdefault("silicon", {})
    dti_cfg = silicon_cfg.setdefault("dti", {})

    if dti_mode != "config":
        dti_cfg["enabled"] = True
        dti_cfg["mode"] = dti_mode
    if dti_mode == "bdti" and bdti_depth is not None:
        dti_cfg["depth"] = float(bdti_depth)


def solver_availability() -> dict[str, bool]:
    names = ["torcwa", "grcwa", "meent", "fmmax", "fdtd", "meep", "jax"]
    availability = {name: importlib.util.find_spec(name) is not None for name in names}
    if availability.get("meep"):
        try:
            import meep as mp

            availability["meep"] = all(
                hasattr(mp, attr)
                for attr in ("Simulation", "Medium", "Vector3", "Block", "PML")
            )
        except ImportError:
            availability["meep"] = False
    return availability


def find_layer_bounds(stack: PixelStack) -> list[tuple[str, float, float]]:
    return [(layer.name, float(layer.z_start), float(layer.z_end)) for layer in stack.layers]


def layer_midpoint(stack: PixelStack, name: str) -> float:
    for layer in stack.layers:
        if layer.name == name:
            return float((layer.z_start + layer.z_end) / 2)
    raise ValueError(f"Layer '{name}' not found")


def dti_summary(stack: PixelStack) -> dict[str, Any]:
    si_cfg = stack._layer_configs.get("silicon", {})
    dti_cfg = si_cfg.get("dti", {})
    si_layer = next((layer for layer in stack.layers if layer.name == "silicon"), None)
    enabled = bool(dti_cfg.get("enabled", False))
    mode = str(dti_cfg.get("mode", "fdti")).lower()
    summary: dict[str, Any] = {
        "enabled": enabled,
        "mode": mode,
        "width_um": float(dti_cfg.get("width", 0.0)),
        "configured_depth_um": float(dti_cfg.get("depth", 0.0)),
    }
    if si_layer is None or not enabled:
        return summary

    if mode == "bdti":
        depth = float(np.clip(dti_cfg.get("depth", si_layer.thickness), 0.0, si_layer.thickness))
        summary["effective_depth_um"] = depth
        summary["z_range_um"] = [float(si_layer.z_end - depth), float(si_layer.z_end)]
    else:
        summary["effective_depth_um"] = float(si_layer.thickness)
        summary["z_range_um"] = [float(si_layer.z_start), float(si_layer.z_end)]
    return summary


def silicon_dti_sample_z(stack: PixelStack) -> float:
    si_layer = next((layer for layer in stack.layers if layer.name == "silicon"), None)
    if si_layer is None:
        return layer_midpoint(stack, "silicon")

    dti_cfg = stack._layer_configs.get("silicon", {}).get("dti", {})
    if dti_cfg.get("enabled", False) and str(dti_cfg.get("mode", "fdti")).lower() == "bdti":
        depth = float(np.clip(dti_cfg.get("depth", si_layer.thickness), 0.0, si_layer.thickness))
        if depth > 0.0:
            return float(si_layer.z_end - depth / 2.0)
    return float((si_layer.z_start + si_layer.z_end) / 2.0)


def build_source_settings(
    stack: PixelStack,
    mode: str,
    reference_pixel: tuple[int, int],
    tilt_deg: float,
) -> SourceSettings:
    rows, cols = stack.unit_cell
    row = int(np.clip(reference_pixel[0], 0, rows - 1))
    col = int(np.clip(reference_pixel[1], 0, cols - 1))
    color = stack.bayer_map[row % len(stack.bayer_map)][col % len(stack.bayer_map[0])]
    active_columns = tuple(range(cols)) if mode == "all-pixels" else (col,)
    return SourceSettings(
        mode=mode,
        reference_pixel=(row, col),
        active_columns=active_columns,
        reference_label=f"{color}_{row}_{col}",
        y_position=(row + 0.5) * stack.pitch,
        tilt_deg=float(tilt_deg),
    )


def build_xz_geometry(
    stack: PixelStack,
    wavelength: float,
    nx: int,
    nz: int,
    y_position: float | None = None,
    n_lens_slices: int = 28,
) -> XZGeometry:
    """Sample the PixelStack into an XZ permittivity cross-section."""

    lx, ly = stack.domain_size
    y_pos = y_position if y_position is not None else 0.5 * stack.pitch
    ny = max(96, nx)
    y_index = int(np.clip(round((y_pos / ly) * (ny - 1)), 0, ny - 1))

    slices = stack.get_layer_slices(
        wavelength,
        nx=nx,
        ny=ny,
        n_lens_slices=n_lens_slices,
    )
    x = np.linspace(0, lx, nx, endpoint=False)
    z_min, z_max = stack.z_range
    z = np.linspace(z_min, z_max, nz)

    eps_xz = np.ones((nz, nx), dtype=complex)
    last_slice = slices[-1]
    for iz, z_val in enumerate(z):
        target = last_slice
        for layer_slice in slices:
            if layer_slice.z_start <= z_val < layer_slice.z_end:
                target = layer_slice
                break
        eps_xz[iz, :] = target.eps_grid[y_index, :]

    boxes = photodiode_boxes_for_xz(stack, y_pos)
    return XZGeometry(
        x=x,
        z=z,
        eps_xz=eps_xz,
        y_position=y_pos,
        layer_bounds=find_layer_bounds(stack),
        photodiode_boxes=boxes,
    )


def eps_xy_at_z(
    stack: PixelStack,
    wavelength: float,
    z_position: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    slices = stack.get_layer_slices(wavelength, nx=nx, ny=ny)
    target = min(
        slices,
        key=lambda s: abs(((s.z_start + s.z_end) / 2.0) - z_position),
    )
    lx, ly = stack.domain_size
    x = np.linspace(0, lx, nx, endpoint=False)
    y = np.linspace(0, ly, ny, endpoint=False)
    return x, y, target.eps_grid, target.name


def photodiode_boxes_for_xz(
    stack: PixelStack,
    y_position: float,
) -> list[tuple[str, float, float, float, float]]:
    """Return photodiode boxes intersecting an XZ plane."""

    si_layer = next((layer for layer in stack.layers if layer.name == "silicon"), None)
    if si_layer is None:
        return []

    row = int(np.clip(math.floor(y_position / stack.pitch), 0, stack.unit_cell[0] - 1))
    boxes: list[tuple[str, float, float, float, float]] = []
    for pd in stack.photodiodes:
        r, c = pd.pixel_index
        if r != row:
            continue
        px, _py, pz = pd.position
        dx, _dy, dz = pd.size
        cx = (c + 0.5) * stack.pitch + px
        cz = si_layer.z_end - pz
        label = f"{pd.color}_{r}_{c}"
        boxes.append((label, cx - dx / 2, cz - dz / 2, dx, dz))
    return boxes


def lens_source_profile(
    x: np.ndarray,
    stack: PixelStack,
    wavelength: float,
    active_columns: tuple[int, ...],
    tilt_deg: float,
) -> np.ndarray:
    """Build a deterministic top illumination profile from microlens centers."""

    field = np.zeros_like(x, dtype=complex)
    k0 = 2.0 * np.pi / wavelength
    focal_length = max(1.2, stack.total_height * 0.42)
    sigma = 0.34 * stack.pitch
    for col in active_columns:
        center = (col + 0.5) * stack.pitch
        aperture = np.exp(-((x - center) ** 2) / (2.0 * sigma**2))
        focus_phase = np.exp(-1j * k0 * ((x - center) ** 2) / (2.0 * focal_length))
        field += aperture * focus_phase
    field *= np.exp(1j * k0 * np.sin(np.deg2rad(tilt_deg)) * (x - x.mean()))
    return field / max(np.max(np.abs(field)), 1e-12)


def angular_spectrum_step(
    field: np.ndarray,
    dx: float,
    dz: float,
    wavelength: float,
    n_eff: float,
) -> np.ndarray:
    freqs = np.fft.fftfreq(field.size, d=dx)
    kx = 2.0 * np.pi * freqs
    k0n = 2.0 * np.pi * max(n_eff, 1e-6) / wavelength
    kz2 = k0n**2 - kx**2
    kz_real = np.sqrt(np.maximum(kz2, 0.0))
    kz_evanescent = np.sqrt(np.maximum(-kz2, 0.0))
    transfer = np.exp(-1j * kz_real * abs(dz)) * np.exp(-kz_evanescent * abs(dz))
    return np.fft.ifft(np.fft.fft(field) * transfer)


def run_rcwa_visual_proxy(
    stack: PixelStack,
    wavelengths: np.ndarray,
    nx: int,
    nz: int,
    source: SourceSettings,
) -> RcwaVisualResult:
    """Run a Fourier-slice propagation visual proxy for RCWA behavior."""

    intensities: dict[float, np.ndarray] = {}
    qe_per_pixel: dict[str, list[float]] = {}
    crosstalk: dict[float, float] = {}
    x_ref: np.ndarray | None = None
    z_ref: np.ndarray | None = None

    t0 = perf_counter()
    for wavelength in wavelengths:
        geom = build_xz_geometry(
            stack,
            wavelength,
            nx=nx,
            nz=nz,
            y_position=source.y_position,
        )
        x_ref = geom.x
        z_ref = geom.z
        dx = float(geom.x[1] - geom.x[0])
        dz = float(geom.z[1] - geom.z[0])
        field = lens_source_profile(
            geom.x,
            stack,
            wavelength,
            source.active_columns,
            source.tilt_deg,
        )
        intensity = np.zeros_like(geom.eps_xz.real)
        k0 = 2.0 * np.pi / wavelength

        for iz in range(nz - 1, -1, -1):
            n_row = np.sqrt(geom.eps_xz[iz, :] + 0j)
            n_eff = float(np.clip(np.nanmean(np.real(n_row)), 1.0, 5.5))
            field = angular_spectrum_step(field, dx, dz, wavelength, n_eff)

            contrast = n_row - np.nanmean(n_row)
            phase = np.exp(-1j * k0 * np.real(contrast) * abs(dz) * 0.20)
            attenuation = np.exp(-np.clip(np.imag(n_row), 0, 6) * k0 * abs(dz) * 0.10)
            field *= phase * attenuation

            metal_like = np.real(geom.eps_xz[iz, :]) > 30
            if np.any(metal_like):
                field[metal_like] *= 0.22

            field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
            max_amp = max(float(np.max(np.abs(field))), 1e-12)
            if max_amp > 8.0:
                field *= 8.0 / max_amp
            intensity[iz, :] = np.abs(field) ** 2

        intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
        intensity /= max(float(np.nanmax(intensity)), 1e-12)
        intensities[float(wavelength)] = intensity
        metrics = integrate_photodiodes(intensity, geom.x, geom.z, geom.photodiode_boxes)
        for name, value in metrics.items():
            qe_per_pixel.setdefault(name, []).append(float(value))
        crosstalk[float(wavelength)] = crosstalk_from_metrics(metrics, source.reference_label)

    if x_ref is None or z_ref is None:
        raise RuntimeError("No wavelengths were simulated")

    elapsed = perf_counter() - t0
    summary = {
        "model": "rcwa_style_fourier_slice_visual_proxy",
        "runtime_seconds": elapsed,
        "note": "Visual surrogate, not a rigorous RCWA backend solve.",
        "source_mode": source.mode,
        "reference_pixel": list(source.reference_pixel),
        "reference_label": source.reference_label,
        "active_columns": list(source.active_columns),
        "tilt_deg": source.tilt_deg,
    }
    return RcwaVisualResult(
        wavelengths=wavelengths,
        x=x_ref,
        z=z_ref,
        intensities=intensities,
        qe_per_pixel=qe_per_pixel,
        crosstalk=crosstalk,
        summary=summary,
    )


def integrate_photodiodes(
    intensity: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    boxes: list[tuple[str, float, float, float, float]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for label, x0, z0, width, height in boxes:
        x_mask = (x >= x0) & (x <= x0 + width)
        z_mask = (z >= z0) & (z <= z0 + height)
        if not np.any(x_mask) or not np.any(z_mask):
            metrics[label] = 0.0
            continue
        window = intensity[np.ix_(z_mask, x_mask)]
        depth_weight = np.linspace(1.2, 0.85, window.shape[0])[:, None]
        metrics[label] = float(np.mean(window * depth_weight))
    return metrics


def crosstalk_from_metrics(metrics: dict[str, float], reference_label: str) -> float:
    total = max(sum(metrics.values()), 1e-12)
    reference = metrics.get(reference_label, 0.0)
    return float(max(total - reference, 0.0) / total)


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=float).ravel()
    b_flat = np.asarray(b, dtype=float).ravel()
    a_flat = a_flat - float(np.mean(a_flat))
    b_flat = b_flat - float(np.mean(b_flat))
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def consistency_metrics(rcwa: RcwaVisualResult, fdtd: FdtdVisualResult) -> dict[str, Any]:
    matched_wavelength = min(rcwa.intensities, key=lambda item: abs(item - fdtd.wavelength))
    rcwa_intensity = rcwa.intensities[matched_wavelength]
    fdtd_intensity = fdtd.average_intensity
    rcwa_center = rcwa_intensity[:, rcwa.x.size // 2]
    fdtd_center = fdtd_intensity[:, fdtd.x.size // 2]
    rcwa_xt = rcwa.crosstalk[matched_wavelength]
    fdtd_xt = fdtd.crosstalk
    gap_pp = float((fdtd_xt - rcwa_xt) * 100.0)
    field_corr = normalized_correlation(rcwa_intensity, fdtd_intensity)
    center_corr = normalized_correlation(rcwa_center, fdtd_center)
    reference_label = str(fdtd.summary.get("reference_label", ""))
    warnings = []
    if abs(gap_pp) > 5.0:
        warnings.append("crosstalk gap exceeds 5 percentage points")
    if field_corr < 0.8:
        warnings.append("field map correlation is below 0.8")
    if abs(float(fdtd.summary.get("energy_tail_relative_change", 0.0))) > 0.10:
        warnings.append("FDTD energy trace has not reached steady state")
    return {
        "matched_wavelength_um": float(matched_wavelength),
        "rcwa_crosstalk": float(rcwa_xt),
        "fdtd_crosstalk": float(fdtd_xt),
        "crosstalk_gap_percentage_points": gap_pp,
        "field_spatial_correlation": field_corr,
        "centerline_correlation": center_corr,
        "fdtd_reference_pd_signal": float(fdtd.qe_per_pixel.get(reference_label, 0.0)),
        "fdtd_energy_tail_relative_change": float(fdtd.summary.get("energy_tail_relative_change", 0.0)),
        "warnings": warnings,
        "note": (
            "These metrics compare matched source, geometry, reference pixel, "
            "normalization, and photodiode integration definitions."
        ),
    }


def run_fdtd_visual_proxy(
    geom: XZGeometry,
    wavelength: float,
    steps: int,
    source_field: np.ndarray,
    source: SourceSettings,
) -> FdtdVisualResult:
    """Run a scalar FDTD-style wave propagation visual proxy."""

    t0 = perf_counter()
    eps_abs = np.maximum(np.abs(np.real(geom.eps_xz)), 1.0)
    n_map = np.clip(np.sqrt(eps_abs), 1.0, 5.5)
    c_map = 1.0 / n_map

    nz, nx = geom.eps_xz.shape
    dx = float(geom.x[1] - geom.x[0])
    dz = float(geom.z[1] - geom.z[0])
    dt = 0.42 / math.sqrt((1.0 / dx**2) + (1.0 / dz**2))
    coeff = (c_map * dt) ** 2

    u_prev = np.zeros((nz, nx), dtype=np.float64)
    u = np.zeros((nz, nx), dtype=np.float64)
    phasor_real = np.zeros_like(u)
    phasor_imag = np.zeros_like(u)
    energy_trace = np.zeros(steps, dtype=np.float64)

    damping_width = max(10, min(nx, nz) // 14)
    source_z = int(np.clip(nz - damping_width - 4, 2, nz - 3))
    source_field = source_field / max(float(np.max(np.abs(source_field))), 1e-12)
    source_real = np.real(source_field).astype(np.float64)
    source_imag = np.imag(source_field).astype(np.float64)

    damping = fdtd_damping_mask(nz, nx, width=damping_width)
    omega = 2.0 * np.pi / wavelength
    snapshot_steps = {
        int(steps * 0.22),
        int(steps * 0.40),
        int(steps * 0.62),
        steps - 1,
    }
    snapshots: list[tuple[int, np.ndarray]] = []
    warmup = int(steps * 0.35)
    phasor_samples = 0

    for step in range(steps):
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
            + (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dz**2
        )
        u_next = 2.0 * u - u_prev + coeff * lap

        ramp_steps = max(8, int(0.18 * steps))
        ramp = min(step / ramp_steps, 1.0)
        envelope = math.sin(0.5 * math.pi * ramp) ** 2
        phase = omega * step * dt
        carrier = source_real * math.cos(phase) + source_imag * math.sin(phase)
        u_next[source_z, :] += 0.20 * envelope * carrier

        u_next *= 1.0 - damping
        u_prev, u = u, u_next
        energy_trace[step] = float(np.mean(u**2))

        if step >= warmup:
            phasor_real += u * math.cos(phase)
            phasor_imag += u * math.sin(phase)
            phasor_samples += 1
        if step in snapshot_steps:
            snapshots.append((step, u.copy()))

    denom = max(1, phasor_samples)
    complex_amplitude = (2.0 / denom) * (phasor_real + 1j * phasor_imag)
    average = np.abs(complex_amplitude) ** 2
    average /= max(float(np.nanmax(average)), 1e-12)
    qe = integrate_photodiodes(average, geom.x, geom.z, geom.photodiode_boxes)
    crosstalk = crosstalk_from_metrics(qe, source.reference_label)
    elapsed = perf_counter() - t0
    tail = energy_trace[int(0.75 * steps):]
    tail_window = max(1, tail.size // 5)
    tail_start = float(np.mean(tail[:tail_window]))
    tail_end = float(np.mean(tail[-tail_window:]))
    tail_mean = max(float(np.mean(tail)), 1e-12)
    energy_tail_relative_change = (tail_end - tail_start) / tail_mean

    summary = {
        "model": "fdtd_style_scalar_wave_visual_proxy",
        "runtime_seconds": elapsed,
        "steps": steps,
        "dt_um_over_c": dt,
        "note": "Visual surrogate with lock-in harmonic extraction, not a rigorous vector FDTD backend solve.",
        "source_mode": source.mode,
        "reference_pixel": list(source.reference_pixel),
        "reference_label": source.reference_label,
        "active_columns": list(source.active_columns),
        "tilt_deg": source.tilt_deg,
        "source_z_index": source_z,
        "harmonic_samples": phasor_samples,
        "energy_tail_relative_change": energy_tail_relative_change,
    }
    return FdtdVisualResult(
        wavelength=wavelength,
        x=geom.x,
        z=geom.z,
        snapshots=snapshots,
        average_intensity=average,
        energy_trace=energy_trace,
        qe_per_pixel={key: float(value) for key, value in qe.items()},
        crosstalk=float(crosstalk),
        summary=summary,
    )


def fdtd_damping_mask(nz: int, nx: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:nz, 0:nx]
    dist = np.minimum.reduce([xx, nx - 1 - xx, yy, nz - 1 - yy]).astype(float)
    ramp = np.clip((width - dist) / max(width, 1), 0.0, 1.0)
    return 0.18 * ramp**2


def plot_structure_overview(
    stack: PixelStack,
    geom: XZGeometry,
    wavelength: float,
    outpath: Path,
) -> None:
    cf_z = layer_midpoint(stack, "color_filter")
    si_z = silicon_dti_sample_z(stack)
    x_cf, y_cf, eps_cf, cf_name = eps_xy_at_z(stack, wavelength, cf_z, 180, 180)
    x_si, y_si, eps_si, si_name = eps_xy_at_z(stack, wavelength, si_z, 180, 180)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(
        np.real(geom.eps_xz),
        extent=[geom.x.min(), geom.x.max(), geom.z.min(), geom.z.max()],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    add_photodiode_boxes(ax, geom.photodiode_boxes)
    add_layer_lines(ax, geom.layer_bounds)
    ax.set_title(f"XZ permittivity, y={geom.y_position:.2f} um, wl={wavelength:.2f} um")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    fig.colorbar(im, ax=ax, label="Re(epsilon)")

    ax = axes[0, 1]
    draw_layer_stack_bar(ax, geom.layer_bounds)

    ax = axes[1, 0]
    im = ax.imshow(
        np.real(eps_cf),
        extent=[x_cf.min(), x_cf.max(), y_cf.min(), y_cf.max()],
        origin="lower",
        aspect="equal",
        cmap="viridis",
    )
    draw_pixel_boundaries(ax, stack)
    ax.set_title(f"XY permittivity at color filter ({cf_name})")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    fig.colorbar(im, ax=ax, label="Re(epsilon)")

    ax = axes[1, 1]
    im = ax.imshow(
        np.real(eps_si),
        extent=[x_si.min(), x_si.max(), y_si.min(), y_si.max()],
        origin="lower",
        aspect="equal",
        cmap="magma",
    )
    draw_pixel_boundaries(ax, stack)
    ax.set_title(f"XY permittivity inside silicon/DTI ({si_name}, z={si_z:.2f} um)")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    fig.colorbar(im, ax=ax, label="Re(epsilon)")

    fig.suptitle("Virtual CMOS pixel geometry used by RCWA/FDTD visual tests", fontsize=15)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_rcwa_visual(result: RcwaVisualResult, geom: XZGeometry, outpath: Path) -> None:
    ncols = len(result.wavelengths)
    fig, axes = plt.subplots(2, ncols, figsize=(5.2 * ncols, 8.2), constrained_layout=True)
    if ncols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, wavelength in enumerate(result.wavelengths):
        intensity = result.intensities[float(wavelength)]
        ax = axes[0, col]
        im = ax.imshow(
            intensity + 1e-6,
            extent=[result.x.min(), result.x.max(), result.z.min(), result.z.max()],
            origin="lower",
            aspect="auto",
            cmap="inferno",
            norm=LogNorm(vmin=1e-4, vmax=1.0),
        )
        add_photodiode_boxes(ax, geom.photodiode_boxes)
        add_layer_lines(ax, geom.layer_bounds, alpha=0.18)
        ax.set_title(f"RCWA-style |E|^2, {wavelength * 1000:.0f} nm")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        fig.colorbar(im, ax=ax, label="normalized |E|^2")

    ax = axes[1, 0]
    wl_nm = result.wavelengths * 1000
    for pixel, values in result.qe_per_pixel.items():
        ax.plot(wl_nm, values, marker="o", linewidth=2, label=pixel)
    ax.set_title("Photodiode collection proxy")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("mean |E|^2 in PD")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1] if ncols > 1 else axes[1, 0]
    xt = [result.crosstalk[float(wl)] * 100.0 for wl in result.wavelengths]
    ax.bar(wl_nm.astype(str), xt, color="#c2410c", alpha=0.85)
    ax.set_title("Neighbor-pixel crosstalk proxy")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("crosstalk (%)")
    ax.grid(True, axis="y", alpha=0.25)

    for col in range(2, ncols):
        ax = axes[1, col]
        centerline = result.intensities[float(result.wavelengths[col])][:, result.x.size // 2]
        ax.plot(result.z, centerline, color="#2563eb")
        ax.set_title(f"Centerline intensity, {result.wavelengths[col] * 1000:.0f} nm")
        ax.set_xlabel("z (um)")
        ax.set_ylabel("normalized |E|^2")
        ax.grid(True, alpha=0.25)

    fig.suptitle("RCWA-style visual optical test", fontsize=15)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_fdtd_visual(result: FdtdVisualResult, geom: XZGeometry, outpath: Path) -> None:
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)

    for idx, (step, snapshot) in enumerate(result.snapshots[:4]):
        ax = fig.add_subplot(gs[0, idx])
        vmax = max(float(np.max(np.abs(snapshot))), 1e-9)
        im = ax.imshow(
            snapshot,
            extent=[result.x.min(), result.x.max(), result.z.min(), result.z.max()],
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        add_layer_lines(ax, geom.layer_bounds, alpha=0.16)
        ax.set_title(f"Wave snapshot step {step}")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        fig.colorbar(im, ax=ax, label="field")

    ax = fig.add_subplot(gs[1:, :2])
    im = ax.imshow(
        result.average_intensity + 1e-6,
        extent=[result.x.min(), result.x.max(), result.z.min(), result.z.max()],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        norm=LogNorm(vmin=1e-4, vmax=1.0),
    )
    add_photodiode_boxes(ax, geom.photodiode_boxes)
    add_layer_lines(ax, geom.layer_bounds, alpha=0.18)
    ax.set_title(f"FDTD-style harmonic |E|^2, {result.wavelength * 1000:.0f} nm")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    fig.colorbar(im, ax=ax, label="normalized |E|^2")

    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(result.energy_trace, color="#0f766e", linewidth=1.7)
    ax.set_title("FDTD visual test energy trace")
    ax.set_xlabel("time step")
    ax.set_ylabel("mean field energy")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2:])
    names = list(result.qe_per_pixel.keys())
    values = [result.qe_per_pixel[name] for name in names]
    ax.bar(names, values, color=["#dc2626", "#16a34a", "#16a34a", "#2563eb"][: len(names)])
    ax.set_title(f"PD collection, crosstalk={result.crosstalk * 100:.1f}%")
    ax.set_ylabel("mean |E|^2 in PD")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("FDTD-style scalar wave visual optical test", fontsize=15)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_comparison(
    rcwa: RcwaVisualResult,
    fdtd: FdtdVisualResult,
    geom: XZGeometry,
    outpath: Path,
) -> None:
    wl = fdtd.wavelength
    rcwa_intensity = rcwa.intensities[min(rcwa.intensities, key=lambda item: abs(item - wl))]
    rcwa_centerline = rcwa_intensity[:, rcwa.x.size // 2]
    fdtd_centerline = fdtd.average_intensity[:, fdtd.x.size // 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(rcwa.z, rcwa_centerline / max(float(rcwa_centerline.max()), 1e-12), label="RCWA-style")
    ax.plot(fdtd.z, fdtd_centerline / max(float(fdtd_centerline.max()), 1e-12), label="FDTD-style")
    ax.set_title("Centerline field comparison")
    ax.set_xlabel("z (um)")
    ax.set_ylabel("normalized |E|^2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    im = ax.imshow(
        np.real(geom.eps_xz),
        extent=[geom.x.min(), geom.x.max(), geom.z.min(), geom.z.max()],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    add_photodiode_boxes(ax, geom.photodiode_boxes)
    ax.set_title("Structure overlay reference")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    fig.colorbar(im, ax=ax, label="Re(epsilon)")

    ax = axes[2]
    rcwa_xt = rcwa.crosstalk[min(rcwa.crosstalk, key=lambda item: abs(item - wl))]
    gap_pp = (fdtd.crosstalk - rcwa_xt) * 100.0
    ax.bar(["RCWA-style", "FDTD-style"], [rcwa_xt * 100, fdtd.crosstalk * 100], color=["#7c3aed", "#0f766e"])
    ax.set_title(f"Crosstalk comparison at {wl * 1000:.0f} nm, gap={gap_pp:+.1f} pp")
    ax.set_ylabel("neighbor crosstalk (%)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Visual RCWA/FDTD comparison dashboard", fontsize=14)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def add_layer_lines(
    ax: plt.Axes,
    layer_bounds: list[tuple[str, float, float]],
    alpha: float = 0.35,
) -> None:
    for _name, z0, z1 in layer_bounds:
        ax.axhline(z0, color="white", linewidth=0.5, alpha=alpha)
        ax.axhline(z1, color="white", linewidth=0.5, alpha=alpha)


def add_photodiode_boxes(
    ax: plt.Axes,
    boxes: list[tuple[str, float, float, float, float]],
) -> None:
    for label, x0, z0, width, height in boxes:
        rect = mpatches.Rectangle(
            (x0, z0),
            width,
            height,
            fill=False,
            edgecolor="#facc15",
            linewidth=1.4,
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            x0 + width / 2,
            z0 + height / 2,
            label,
            ha="center",
            va="center",
            color="#facc15",
            fontsize=7,
            weight="bold",
        )


def draw_pixel_boundaries(ax: plt.Axes, stack: PixelStack) -> None:
    rows, cols = stack.unit_cell
    for c in range(cols + 1):
        ax.axvline(c * stack.pitch, color="white", linewidth=0.6, linestyle="--", alpha=0.7)
    for r in range(rows + 1):
        ax.axhline(r * stack.pitch, color="white", linewidth=0.6, linestyle="--", alpha=0.7)


def draw_layer_stack_bar(ax: plt.Axes, layer_bounds: list[tuple[str, float, float]]) -> None:
    colors = plt.get_cmap("tab20")
    for idx, (_name, z0, z1) in enumerate(layer_bounds):
        ax.add_patch(
            mpatches.Rectangle((0.0, z0), 1.0, z1 - z0, facecolor=colors(idx), alpha=0.75)
        )
    labels: list[tuple[str, float, float]] = []
    idx = 0
    while idx < len(layer_bounds):
        name, z0, z1 = layer_bounds[idx]
        if name.startswith("barl_"):
            start = z0
            end = z1
            idx += 1
            while idx < len(layer_bounds) and layer_bounds[idx][0].startswith("barl_"):
                end = layer_bounds[idx][2]
                idx += 1
            labels.append(("BARL stack", start, end))
        else:
            labels.append((name, z0, z1))
            idx += 1

    for name, z0, z1 in labels:
        ax.text(1.08, (z0 + z1) / 2, f"{name} ({z1 - z0:.3f} um)", va="center", fontsize=8)
    z_min = min(z0 for _name, z0, _z1 in layer_bounds)
    z_max = max(z1 for _name, _z0, z1 in layer_bounds)
    ax.set_xlim(0, 2.4)
    ax.set_ylim(z_min, z_max)
    ax.set_xticks([])
    ax.set_ylabel("z (um)")
    ax.set_title("Layer stack thickness map")
    ax.grid(True, axis="y", alpha=0.25)


def write_report(
    outdir: Path,
    config_path: Path,
    dti_info: dict[str, Any],
    availability: dict[str, bool],
    rcwa: RcwaVisualResult,
    fdtd: FdtdVisualResult,
    consistency: dict[str, Any],
    plots: dict[str, str],
) -> None:
    metrics = {
        "config": str(config_path),
        "dti": dti_info,
        "solver_backend_availability": availability,
        "rcwa_visual": {
            **rcwa.summary,
            "wavelengths_um": rcwa.wavelengths.tolist(),
            "qe_per_pixel": rcwa.qe_per_pixel,
            "crosstalk": {str(k): v for k, v in rcwa.crosstalk.items()},
        },
        "fdtd_visual": {
            **fdtd.summary,
            "wavelength_um": fdtd.wavelength,
            "qe_per_pixel": fdtd.qe_per_pixel,
            "crosstalk": fdtd.crosstalk,
        },
        "consistency": consistency,
        "plots": plots,
    }
    (outdir / "visual_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# CMOS Pixel Visual Optical Simulation Report",
        "",
        f"- Config: `{config_path}`",
        f"- DTI: `{dti_info}`",
        f"- RCWA-style model: `{rcwa.summary['model']}`",
        f"- FDTD-style model: `{fdtd.summary['model']}`",
        f"- Source mode: `{rcwa.summary['source_mode']}`, reference: `{rcwa.summary['reference_label']}`",
        "- Backend availability: "
        + ", ".join(f"{name}={available}" for name, available in availability.items()),
        "",
        "These plots are visual test artifacts. They show whether the pixel geometry,",
        "field confinement, photodiode collection, and crosstalk behavior look plausible.",
        "The lightweight models are not a replacement for rigorous backend sign-off.",
        "",
        "Alignment controls used here:",
        "- Crosstalk uses single reference-pixel illumination by default.",
        "- RCWA and FDTD share the same reference pixel, source profile, tilt, and PD integration.",
        "- FDTD uses harmonic lock-in extraction so it is compared against a frequency-domain field.",
        "- If `source-mode=all-pixels`, neighbor signal is direct illumination, not pure crosstalk.",
        "",
    ]
    for title, filename in plots.items():
        lines.extend([f"## {title}", "", f"![{title}]({filename})", ""])
    lines.append("## Metrics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(metrics["consistency"], indent=2))
    lines.append("```")
    (outdir / "visual_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.nx = min(args.nx, 110)
        args.nz = min(args.nz, 160)
        args.fdtd_steps = min(args.fdtd_steps, 900)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    stack = load_pixel_stack(args.config, args.dti_mode, args.bdti_depth)
    wavelengths = np.array(args.wavelengths, dtype=float)
    source = build_source_settings(
        stack,
        args.source_mode,
        tuple(args.reference_pixel),
        args.tilt_deg,
    )
    geom = build_xz_geometry(
        stack,
        args.fdtd_wavelength,
        nx=args.nx,
        nz=args.nz,
        y_position=source.y_position,
    )
    availability = solver_availability()
    dti_info = dti_summary(stack)

    source_field = lens_source_profile(
        geom.x,
        stack,
        args.fdtd_wavelength,
        source.active_columns,
        source.tilt_deg,
    )
    rcwa = run_rcwa_visual_proxy(stack, wavelengths, nx=args.nx, nz=args.nz, source=source)
    fdtd = run_fdtd_visual_proxy(
        geom,
        args.fdtd_wavelength,
        steps=args.fdtd_steps,
        source_field=source_field,
        source=source,
    )
    consistency = consistency_metrics(rcwa, fdtd)

    plots = {
        "Geometry overview": "01_geometry_overview.png",
        "RCWA-style visual test": "02_rcwa_visual_test.png",
        "FDTD-style visual test": "03_fdtd_visual_test.png",
        "RCWA/FDTD comparison": "04_rcwa_fdtd_comparison.png",
    }
    plot_structure_overview(stack, geom, args.fdtd_wavelength, outdir / plots["Geometry overview"])
    plot_rcwa_visual(rcwa, geom, outdir / plots["RCWA-style visual test"])
    plot_fdtd_visual(fdtd, geom, outdir / plots["FDTD-style visual test"])
    plot_comparison(rcwa, fdtd, geom, outdir / plots["RCWA/FDTD comparison"])
    write_report(outdir, args.config, dti_info, availability, rcwa, fdtd, consistency, plots)

    print(f"Visual optical simulation artifacts written to: {outdir}")
    for filename in plots.values():
        print(f"  - {outdir / filename}")
    print(f"  - {outdir / 'visual_metrics.json'}")
    print(f"  - {outdir / 'visual_report.md'}")


if __name__ == "__main__":
    main()
