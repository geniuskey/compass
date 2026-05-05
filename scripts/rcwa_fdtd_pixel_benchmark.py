#!/usr/bin/env python3
"""2x2 Bayer pixel RCWA/FDTD visual benchmark.

This is the rung after the 2D periodic trench benchmark. It uses the real
PixelStack geometry path for a 2x2 Bayer cell, including FDTI/BDTI silicon
slices, material-database complex permittivity, microlens staircasing, color
filters, BARL layers, and photodiode regions.

The outputs are intentionally visual and Python-level:

* torcwa RCWA R/T/A for the full 2x2 periodic pixel supercell.
* 3D scalar time-domain FDTD localized source scans.
* Photodiode absorption proxy and source-to-PD crosstalk matrices.
* Geometry, field, crosstalk, and per-PD collection plots.

The FDTD crosstalk path is scalar, not a full-vector production FDTD solver.
It is useful as a visually inspectable, higher-fidelity benchmark before
building a GUI or spending much longer runs on a production backend.
"""

from __future__ import annotations

import argparse
import copy
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
from matplotlib.colors import LogNorm
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.geometry.pixel_stack import PixelStack  # noqa: E402
from compass.materials.database import MaterialDB  # noqa: E402


@dataclass(frozen=True)
class PixelSource:
    label: str
    row: int
    col: int
    color: str
    x_um: float
    y_um: float


@dataclass
class RcwaRta:
    reflection: float
    transmission: float
    absorption: float
    order_used: int
    fallback_errors: list[dict[str, str]]
    runtime_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "R": self.reflection,
            "T": self.transmission,
            "A": self.absorption,
            "R_plus_T_plus_A": self.reflection + self.transmission + self.absorption,
            "order_used": self.order_used,
            "fallback_errors": self.fallback_errors,
            "runtime_s": self.runtime_s,
        }


@dataclass
class FdtdScan:
    source: PixelSource
    pd_signals: dict[str, float]
    pd_fractions: dict[str, float]
    total_pd_signal: float
    intensity: np.ndarray
    loss_density: np.ndarray
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2x2 Bayer pixel RCWA/FDTD visual benchmark.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pixel" / "default_bsi_1um.yaml",
        help="Pixel YAML config.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "rcwa_fdtd_pixel_benchmark",
        help="Output directory for plots and report files.",
    )
    parser.add_argument(
        "--dti-modes",
        choices=["fdti", "bdti"],
        nargs="+",
        default=["fdti", "bdti"],
        help="DTI modes to evaluate.",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=0.55,
        help="Primary wavelength in um.",
    )
    parser.add_argument(
        "--bdti-depth",
        type=float,
        default=0.60,
        help="BDTI trench depth in um.",
    )
    parser.add_argument("--nx", type=int, default=44, help="FDTD/geometry x samples.")
    parser.add_argument("--ny", type=int, default=44, help="FDTD/geometry y samples.")
    parser.add_argument("--nz", type=int, default=118, help="FDTD/geometry z samples.")
    parser.add_argument(
        "--fdtd-steps",
        type=int,
        default=950,
        help="3D scalar FDTD timesteps per localized source.",
    )
    parser.add_argument(
        "--source-waist",
        type=float,
        default=0.32,
        help="Gaussian source waist in um.",
    )
    parser.add_argument(
        "--source-set",
        choices=["all", "diagonal", "single"],
        default="all",
        help="Which source pixels to scan.",
    )
    parser.add_argument(
        "--reference-pixel",
        type=int,
        nargs=2,
        default=[0, 0],
        metavar=("ROW", "COL"),
        help="Source pixel for --source-set single.",
    )
    parser.add_argument(
        "--rcwa-order",
        type=int,
        default=1,
        help="Preferred torcwa order for the full 2x2 supercell.",
    )
    parser.add_argument("--rcwa-nx", type=int, default=72, help="RCWA x samples.")
    parser.add_argument("--rcwa-ny", type=int, default=72, help="RCWA y samples.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer sources and coarser FDTD settings for smoke checks.",
    )
    return parser.parse_args()


def load_config(config_path: Path, dti_mode: str, bdti_depth: float) -> dict[str, Any]:
    raw = OmegaConf.load(config_path)
    config = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping config from {config_path}")
    cfg = copy.deepcopy(config)
    silicon = cfg.setdefault("pixel", {}).setdefault("layers", {}).setdefault("silicon", {})
    dti = silicon.setdefault("dti", {})
    dti["enabled"] = True
    dti["mode"] = dti_mode
    if dti_mode == "bdti":
        dti["depth"] = float(bdti_depth)
    return cfg


def pixel_sources(stack: PixelStack) -> list[PixelSource]:
    sources: list[PixelSource] = []
    rows, cols = stack.unit_cell
    for r in range(rows):
        for c in range(cols):
            color = stack.bayer_map[r % len(stack.bayer_map)][c % len(stack.bayer_map[0])]
            sources.append(
                PixelSource(
                    label=f"{color}_{r}_{c}",
                    row=r,
                    col=c,
                    color=color,
                    x_um=(c + 0.5) * stack.pitch,
                    y_um=(r + 0.5) * stack.pitch,
                )
            )
    return sources


def select_sources(sources: list[PixelSource], args: argparse.Namespace) -> list[PixelSource]:
    if args.quick:
        return [src for src in sources if (src.row, src.col) in {(0, 0), (1, 1)}]
    if args.source_set == "all":
        return sources
    if args.source_set == "diagonal":
        return [src for src in sources if src.row == src.col]
    row, col = args.reference_pixel
    return [src for src in sources if src.row == row and src.col == col]


def dti_info(stack: PixelStack) -> dict[str, Any]:
    si_layer = next(layer for layer in stack.layers if layer.name == "silicon")
    dti = stack._layer_configs.get("silicon", {}).get("dti", {})
    mode = str(dti.get("mode", "fdti")).lower()
    depth = si_layer.thickness if mode == "fdti" else float(dti.get("depth", si_layer.thickness))
    return {
        "enabled": bool(dti.get("enabled", False)),
        "mode": mode,
        "width_um": float(dti.get("width", 0.0)),
        "effective_depth_um": float(np.clip(depth, 0.0, si_layer.thickness)),
        "silicon_z_range_um": [float(si_layer.z_start), float(si_layer.z_end)],
    }


def run_torcwa_pixel_rta(
    stack: PixelStack,
    wavelength_um: float,
    preferred_order: int,
    nx: int,
    ny: int,
) -> RcwaRta:
    errors: list[dict[str, str]] = []
    for order in range(preferred_order, -1, -1):
        try:
            return run_torcwa_pixel_rta_once(stack, wavelength_um, order, nx, ny, errors)
        except Exception as exc:
            errors.append({"order": str(order), "error": repr(exc)})
    raise RuntimeError(f"torcwa failed for all orders: {errors}")


def run_torcwa_pixel_rta_once(
    stack: PixelStack,
    wavelength_um: float,
    order: int,
    nx: int,
    ny: int,
    fallback_errors: list[dict[str, str]],
) -> RcwaRta:
    t0 = perf_counter()
    import torch
    import torcwa

    lx, ly = stack.domain_size
    slices = stack.get_layer_slices(wavelength_um, nx=nx, ny=ny, n_lens_slices=10)
    sim = torcwa.rcwa(
        freq=1.0 / wavelength_um,
        order=[order, order],
        L=[lx, ly],
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(0.0, 0.0)
    for layer_slice in reversed(slices):
        eps_tensor = torch.tensor(layer_slice.eps_grid.T.copy(), dtype=torch.complex64)
        sim.add_layer(thickness=layer_slice.thickness, eps=eps_tensor)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[0.0, 1.0], direction="forward", notation="xy")
    r = diffraction_power_sum(sim, torch, order, port="reflection")
    t = diffraction_power_sum(sim, torch, order, port="transmission")
    return RcwaRta(
        reflection=float(r),
        transmission=float(t),
        absorption=float(1.0 - r - t),
        order_used=order,
        fallback_errors=list(fallback_errors),
        runtime_s=perf_counter() - t0,
    )


def diffraction_power_sum(sim: Any, torch: Any, order: int, port: str) -> float:
    total = 0.0
    for mx in range(-order, order + 1):
        for my in range(-order, order + 1):
            try:
                coeff = sim.S_parameters(
                    orders=[mx, my],
                    direction="forward",
                    port=port,
                    polarization="yy",
                    power_norm=True,
                )
            except Exception:
                continue
            value = float(torch.abs(coeff) ** 2)
            if np.isfinite(value):
                total += max(value, 0.0)
    return total


def build_eps_zyx(
    stack: PixelStack,
    wavelength_um: float,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Sample PixelStack to top-down (z, y, x) permittivity."""

    slices = stack.get_layer_slices(wavelength_um, nx=nx, ny=ny, n_lens_slices=14)
    x = np.linspace(0.0, stack.domain_size[0], nx, endpoint=False) + 0.5 * stack.domain_size[0] / nx
    y = np.linspace(0.0, stack.domain_size[1], ny, endpoint=False) + 0.5 * stack.domain_size[1] / ny
    z_bottom = np.linspace(stack.z_range[0], stack.z_range[1], nz, endpoint=False)
    z_bottom = z_bottom + 0.5 * (stack.z_range[1] - stack.z_range[0]) / nz

    eps_bottom = np.ones((nz, ny, nx), dtype=np.complex128)
    layer_bounds = []
    for layer_slice in slices:
        mask = (z_bottom >= layer_slice.z_start) & (z_bottom < layer_slice.z_end)
        eps_bottom[mask, :, :] = layer_slice.eps_grid[np.newaxis, :, :]
        layer_bounds.append(
            {
                "name": layer_slice.name,
                "z_start_um": float(layer_slice.z_start),
                "z_end_um": float(layer_slice.z_end),
                "thickness_um": float(layer_slice.thickness),
            }
        )

    eps_top = eps_bottom[::-1, :, :]
    depth = stack.z_range[1] - z_bottom[::-1]
    return x, y, depth, eps_top, layer_bounds


def pad_z_with_air(
    eps_zyx: np.ndarray,
    depth_um: np.ndarray,
    dz_um: float,
    pml_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    eps_air = 1.0 + 0j
    padded = np.pad(
        eps_zyx,
        ((pml_cells, pml_cells), (0, 0), (0, 0)),
        mode="constant",
        constant_values=eps_air,
    )
    full_depth = (np.arange(padded.shape[0]) + 0.5) * dz_um - pml_cells * dz_um
    return padded, full_depth


def run_scalar_fdtd_scan(
    stack: PixelStack,
    wavelength_um: float,
    source: PixelSource,
    nx: int,
    ny: int,
    nz: int,
    steps: int,
    source_waist_um: float,
) -> FdtdScan:
    t0 = perf_counter()
    x, y, depth, eps_zyx, layer_bounds = build_eps_zyx(stack, wavelength_um, nx, ny, nz)
    dz_um = float(np.mean(np.diff(depth)))
    dx_um = float(stack.domain_size[0] / nx)
    dy_um = float(stack.domain_size[1] / ny)
    pml_cells = max(14, round(0.55 / dz_um))
    eps_full, depth_full = pad_z_with_air(eps_zyx, depth, dz_um, pml_cells)

    nz_full = eps_full.shape[0]
    source_z = pml_cells + max(3, round(0.18 / dz_um))
    dt = 0.38 / math.sqrt((1.0 / dx_um**2) + (1.0 / dy_um**2) + (1.0 / dz_um**2))
    omega = 2.0 * np.pi / wavelength_um

    eps_real = np.maximum(np.real(eps_full), 1.0)
    sigma = np.maximum(np.imag(eps_full), 0.0) * omega
    c2dt2 = (dt / np.sqrt(eps_real)) ** 2
    material_damp = np.exp(-0.5 * sigma * dt / eps_real)
    pml_damp = fdtd_z_damping(nz_full, pml_cells)[:, np.newaxis, np.newaxis]

    profile = source_profile_xy(x, y, stack.domain_size, source, source_waist_um)
    u_prev = np.zeros_like(eps_full.real)
    u = np.zeros_like(eps_full.real)
    phasor = np.zeros_like(eps_full, dtype=np.complex128)
    energy_trace = np.zeros(steps, dtype=np.float64)
    warmup = int(0.45 * steps)
    ramp_steps = max(20, int(0.16 * steps))
    samples = 0

    for step in range(steps):
        lap = np.zeros_like(u)
        center = u[1:-1, :, :]
        lap[1:-1, :, :] = (
            (np.roll(center, -1, axis=2) - 2.0 * center + np.roll(center, 1, axis=2)) / dx_um**2
            + (np.roll(center, -1, axis=1) - 2.0 * center + np.roll(center, 1, axis=1)) / dy_um**2
            + (u[2:, :, :] - 2.0 * center + u[:-2, :, :]) / dz_um**2
        )
        u_next = 2.0 * u - u_prev + c2dt2 * lap

        ramp = 1.0
        if step < ramp_steps:
            ramp = np.sin(0.5 * np.pi * step / ramp_steps) ** 2
        phase = omega * step * dt
        u_next[source_z, :, :] += 0.22 * ramp * np.sin(phase) * profile

        u_next *= material_damp
        u_next *= pml_damp
        u_prev, u = u, u_next
        energy_trace[step] = float(np.mean(u * u))

        if step >= warmup:
            phasor += u * np.exp(-1j * phase)
            samples += 1

    complex_amp = (2.0 / max(samples, 1)) * phasor
    intensity = np.abs(complex_amp) ** 2
    loss_density = sigma * intensity
    pd_signals = integrate_pd_signals(stack, x, y, depth_full, loss_density)
    total_pd = max(sum(pd_signals.values()), 1e-30)
    pd_fractions = {label: float(value / total_pd) for label, value in pd_signals.items()}
    max_intensity = max(float(np.max(intensity)), 1e-30)
    intensity_norm = intensity / max_intensity

    tail = energy_trace[int(0.75 * steps):]
    tail_change = 0.0
    if tail.size > 4:
        window = max(1, tail.size // 5)
        tail_change = float((np.mean(tail[-window:]) - np.mean(tail[:window])) / max(np.mean(tail), 1e-30))

    return FdtdScan(
        source=source,
        pd_signals={key: float(value) for key, value in pd_signals.items()},
        pd_fractions=pd_fractions,
        total_pd_signal=float(total_pd),
        intensity=intensity_norm,
        loss_density=loss_density,
        metadata={
            "runtime_s": perf_counter() - t0,
            "nx": nx,
            "ny": ny,
            "nz_struct": nz,
            "nz_full": nz_full,
            "dx_um": dx_um,
            "dy_um": dy_um,
            "dz_um": dz_um,
            "dt_um_over_c": dt,
            "steps": steps,
            "source_z": source_z,
            "pml_cells": pml_cells,
            "harmonic_samples": samples,
            "energy_tail_relative_change": tail_change,
            "layer_bounds": layer_bounds,
        },
    )


def source_profile_xy(
    x: np.ndarray,
    y: np.ndarray,
    domain_size: tuple[float, float],
    source: PixelSource,
    waist_um: float,
) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="xy")
    lx, ly = domain_size
    dx = np.abs(xx - source.x_um)
    dy = np.abs(yy - source.y_um)
    dx = np.minimum(dx, lx - dx)
    dy = np.minimum(dy, ly - dy)
    profile = np.exp(-(dx**2 + dy**2) / (2.0 * waist_um**2))
    return profile / max(float(np.max(profile)), 1e-30)


def fdtd_z_damping(nz: int, pml_cells: int) -> np.ndarray:
    damping = np.ones(nz, dtype=np.float64)
    for i in range(pml_cells):
        strength = ((pml_cells - i) / pml_cells) ** 3
        value = np.exp(-0.10 * strength)
        damping[i] = min(damping[i], value)
        damping[-1 - i] = min(damping[-1 - i], value)
    return damping


def integrate_pd_signals(
    stack: PixelStack,
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    loss_density: np.ndarray,
) -> dict[str, float]:
    si_layer = next(layer for layer in stack.layers if layer.name == "silicon")
    total_height = stack.z_range[1]
    signals: dict[str, float] = {}
    for pd in stack.photodiodes:
        r, c = pd.pixel_index
        label = f"{pd.color}_{r}_{c}"
        px, py, pz_from_si_top = pd.position
        sx, sy, sz = pd.size
        cx = (c + 0.5) * stack.pitch + px
        cy = (r + 0.5) * stack.pitch + py
        physical_z = si_layer.z_end - pz_from_si_top
        cdepth = total_height - physical_z

        x_mask = periodic_box_mask(x, cx, sx, stack.domain_size[0])
        y_mask = periodic_box_mask(y, cy, sy, stack.domain_size[1])
        z_mask = (depth >= cdepth - sz / 2.0) & (depth <= cdepth + sz / 2.0)
        if not np.any(x_mask) or not np.any(y_mask) or not np.any(z_mask):
            signals[label] = 0.0
            continue
        region = loss_density[np.ix_(z_mask, y_mask, x_mask)]
        signals[label] = float(np.sum(region))
    return signals


def periodic_box_mask(values: np.ndarray, center: float, width: float, period: float) -> np.ndarray:
    dist = np.abs(values - center)
    dist = np.minimum(dist, period - dist)
    return dist <= width / 2.0


def crosstalk_matrix(scans: list[FdtdScan], labels: list[str]) -> np.ndarray:
    matrix = np.zeros((len(scans), len(labels)), dtype=float)
    for i, scan in enumerate(scans):
        for j, label in enumerate(labels):
            matrix[i, j] = scan.pd_fractions.get(label, 0.0)
    return matrix


def summarize_mode(rcwa: RcwaRta, scans: list[FdtdScan], labels: list[str]) -> dict[str, Any]:
    mat = crosstalk_matrix(scans, labels)
    diagonal = []
    off_diagonal = []
    for i, scan in enumerate(scans):
        if scan.source.label in labels:
            j = labels.index(scan.source.label)
            diagonal.append(mat[i, j])
            off_diagonal.extend([mat[i, k] for k in range(mat.shape[1]) if k != j])
    max_tail_change = float(
        np.max([abs(scan.metadata["energy_tail_relative_change"]) for scan in scans])
    )
    warnings = []
    if max_tail_change > 0.15:
        warnings.append("FDTD energy tail changed by more than 15%; increase --fdtd-steps for final runs.")
    if rcwa.fallback_errors:
        warnings.append("RCWA used a lower fallback order for this pixel supercell.")
    return {
        "rcwa_rta": rcwa.as_dict(),
        "mean_self_collection_fraction": float(np.mean(diagonal)) if diagonal else 0.0,
        "max_neighbor_crosstalk_fraction": float(np.max(off_diagonal)) if off_diagonal else 0.0,
        "mean_total_pd_signal": float(np.mean([scan.total_pd_signal for scan in scans])),
        "max_energy_tail_relative_change": max_tail_change,
        "warnings": warnings,
    }


def plot_geometry(
    mode_payloads: dict[str, dict[str, Any]],
    outpath: Path,
) -> None:
    n_modes = len(mode_payloads)
    fig, axes = plt.subplots(n_modes, 2, figsize=(11, 4.5 * n_modes), constrained_layout=True)
    if n_modes == 1:
        axes = np.asarray([axes])
    for row_axes, (mode, payload) in zip(axes, mode_payloads.items()):
        stack: PixelStack = payload["stack"]
        wavelength = payload["wavelength_um"]
        x, y, depth, eps, _bounds = build_eps_zyx(stack, wavelength, 96, 96, 110)
        cf_idx = int(np.argmin(np.abs(depth - 1.25)))
        si_idx = int(np.argmin(np.abs(depth - 3.2)))
        for ax, idx, title in [
            (row_axes[0], cf_idx, "color/filter stack"),
            (row_axes[1], si_idx, "silicon DTI region"),
        ]:
            im = ax.imshow(
                np.real(eps[idx]),
                extent=[x[0], x[-1], y[0], y[-1]],
                origin="lower",
                cmap="viridis",
                aspect="equal",
            )
            ax.set_title(f"{mode.upper()} {title}, Re(eps)")
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_rcwa_rta(mode_payloads: dict[str, dict[str, Any]], outpath: Path) -> None:
    modes = list(mode_payloads)
    keys = ["R", "T", "A"]
    data = np.array([[mode_payloads[mode]["rcwa"].as_dict()[key] for key in keys] for mode in modes])
    x = np.arange(len(modes))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for i, key in enumerate(keys):
        ax.bar(x + (i - 1) * width, data[:, i], width=width, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels([mode.upper() for mode in modes])
    ax.set_ylabel("Power fraction")
    ax.set_title("torcwa RCWA R/T/A for 2x2 Bayer supercell")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_crosstalk_matrices(
    mode_payloads: dict[str, dict[str, Any]],
    labels: list[str],
    outpath: Path,
) -> None:
    n_modes = len(mode_payloads)
    fig, axes = plt.subplots(1, n_modes, figsize=(6.2 * n_modes, 5.4), constrained_layout=True)
    if n_modes == 1:
        axes = [axes]
    for ax, (mode, payload) in zip(axes, mode_payloads.items()):
        scans: list[FdtdScan] = payload["scans"]
        source_labels = [scan.source.label for scan in scans]
        matrix = crosstalk_matrix(scans, labels)
        im = ax.imshow(matrix, vmin=0.0, vmax=max(0.5, float(matrix.max())), cmap="magma")
        ax.set_title(f"{mode.upper()} FDTD PD collection fractions")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(source_labels)))
        ax.set_yticklabels(source_labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_field_slices(
    mode_payloads: dict[str, dict[str, Any]],
    outpath: Path,
) -> None:
    n_modes = len(mode_payloads)
    fig, axes = plt.subplots(n_modes, 2, figsize=(11, 4.4 * n_modes), constrained_layout=True)
    if n_modes == 1:
        axes = np.asarray([axes])
    for row_axes, (mode, payload) in zip(axes, mode_payloads.items()):
        scan: FdtdScan = payload["scans"][0]
        stack: PixelStack = payload["stack"]
        nx = scan.intensity.shape[2]
        ny = scan.intensity.shape[1]
        nz = scan.intensity.shape[0]
        x = np.linspace(0.0, stack.domain_size[0], nx, endpoint=False)
        y = np.linspace(0.0, stack.domain_size[1], ny, endpoint=False)
        depth = (np.arange(nz) + 0.5) * scan.metadata["dz_um"] - scan.metadata["pml_cells"] * scan.metadata["dz_um"]
        y_idx = int(np.clip(round((scan.source.y_um / stack.domain_size[1]) * (ny - 1)), 0, ny - 1))
        z_pd = int(np.argmax(np.sum(scan.loss_density, axis=(1, 2))))

        im0 = row_axes[0].imshow(
            np.maximum(scan.intensity[:, y_idx, :], 1e-6),
            extent=[x[0], x[-1], depth[-1], depth[0]],
            aspect="auto",
            norm=LogNorm(vmin=1e-5, vmax=1.0),
            cmap="magma",
        )
        row_axes[0].set_title(f"{mode.upper()} XZ field, source {scan.source.label}")
        row_axes[0].set_xlabel("x (um)")
        row_axes[0].set_ylabel("depth from stack top (um)")
        fig.colorbar(im0, ax=row_axes[0], fraction=0.046, pad=0.04)

        im1 = row_axes[1].imshow(
            np.maximum(scan.intensity[z_pd], 1e-6),
            extent=[x[0], x[-1], y[0], y[-1]],
            origin="lower",
            norm=LogNorm(vmin=1e-5, vmax=1.0),
            cmap="magma",
            aspect="equal",
        )
        row_axes[1].set_title(f"{mode.upper()} XY field near peak PD loss")
        row_axes[1].set_xlabel("x (um)")
        row_axes[1].set_ylabel("y (um)")
        fig.colorbar(im1, ax=row_axes[1], fraction=0.046, pad=0.04)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def write_report(
    outdir: Path,
    args: argparse.Namespace,
    summary: dict[str, Any],
    plots: dict[str, str],
) -> None:
    lines = [
        "# 2x2 Bayer Pixel RCWA/FDTD Benchmark",
        "",
        f"- Config: `{args.config}`",
        f"- Wavelength: `{args.wavelength}` um",
        f"- DTI modes: `{', '.join(args.dti_modes)}`",
        f"- BDTI depth: `{args.bdti_depth}` um",
        f"- FDTD grid: `{args.nx} x {args.ny} x {args.nz}`, steps `{args.fdtd_steps}`",
        f"- RCWA order: `{args.rcwa_order}`",
        "",
        "RCWA reports full-supercell R/T/A. FDTD reports localized-source",
        "photodiode loss-density fractions and crosstalk matrices.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]
    for title, filename in plots.items():
        lines.extend([f"## {title}", "", f"![{title}]({filename})", ""])
    (outdir / "pixel_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.nx = min(args.nx, 34)
        args.ny = min(args.ny, 34)
        args.nz = min(args.nz, 90)
        args.fdtd_steps = min(args.fdtd_steps, 520)
        args.rcwa_nx = min(args.rcwa_nx, 48)
        args.rcwa_ny = min(args.rcwa_ny, 48)
        args.source_set = "diagonal"

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    db = MaterialDB()

    mode_payloads: dict[str, dict[str, Any]] = {}
    labels: list[str] | None = None
    for mode in args.dti_modes:
        config = load_config(args.config, mode, args.bdti_depth)
        stack = PixelStack(config, db)
        sources = pixel_sources(stack)
        labels = [source.label for source in sources]
        selected = select_sources(sources, args)
        rcwa = run_torcwa_pixel_rta(
            stack,
            wavelength_um=args.wavelength,
            preferred_order=args.rcwa_order,
            nx=args.rcwa_nx,
            ny=args.rcwa_ny,
        )
        scans = [
            run_scalar_fdtd_scan(
                stack,
                wavelength_um=args.wavelength,
                source=source,
                nx=args.nx,
                ny=args.ny,
                nz=args.nz,
                steps=args.fdtd_steps,
                source_waist_um=args.source_waist,
            )
            for source in selected
        ]
        mode_payloads[mode] = {
            "stack": stack,
            "wavelength_um": args.wavelength,
            "dti": dti_info(stack),
            "sources": selected,
            "rcwa": rcwa,
            "scans": scans,
        }

    assert labels is not None
    summary = {
        mode: {
            "dti": payload["dti"],
            **summarize_mode(payload["rcwa"], payload["scans"], labels),
        }
        for mode, payload in mode_payloads.items()
    }
    metrics = {
        "settings": {
            "config": str(args.config),
            "wavelength_um": args.wavelength,
            "dti_modes": args.dti_modes,
            "bdti_depth_um": args.bdti_depth,
            "fdtd_grid": [args.nx, args.ny, args.nz],
            "fdtd_steps": args.fdtd_steps,
            "source_waist_um": args.source_waist,
            "rcwa_order": args.rcwa_order,
            "rcwa_grid": [args.rcwa_nx, args.rcwa_ny],
            "source_set": args.source_set,
        },
        "labels": labels,
        "summary": summary,
        "crosstalk_matrices": {
            mode: crosstalk_matrix(payload["scans"], labels).tolist()
            for mode, payload in mode_payloads.items()
        },
        "source_rows": {
            mode: [scan.source.label for scan in payload["scans"]]
            for mode, payload in mode_payloads.items()
        },
        "pd_signals": {
            mode: {
                scan.source.label: {
                    "signals": scan.pd_signals,
                    "fractions": scan.pd_fractions,
                    "total_pd_signal": scan.total_pd_signal,
                    "metadata": scan.metadata,
                }
                for scan in payload["scans"]
            }
            for mode, payload in mode_payloads.items()
        },
    }
    (outdir / "pixel_benchmark_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    plots = {
        "Geometry Slices": "01_geometry_slices.png",
        "RCWA RTA": "02_rcwa_rta.png",
        "FDTD Crosstalk Matrix": "03_fdtd_crosstalk_matrix.png",
        "FDTD Field Slices": "04_fdtd_field_slices.png",
    }
    plot_geometry(mode_payloads, outdir / plots["Geometry Slices"])
    plot_rcwa_rta(mode_payloads, outdir / plots["RCWA RTA"])
    plot_crosstalk_matrices(mode_payloads, labels, outdir / plots["FDTD Crosstalk Matrix"])
    plot_field_slices(mode_payloads, outdir / plots["FDTD Field Slices"])
    write_report(outdir, args, summary, plots)

    print(f"2x2 Bayer pixel benchmark artifacts written to: {outdir}")
    print(f"  - {outdir / 'pixel_benchmark_metrics.json'}")
    print(f"  - {outdir / 'pixel_benchmark_report.md'}")
    for filename in plots.values():
        print(f"  - {outdir / filename}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
