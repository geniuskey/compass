#!/usr/bin/env python3
"""2D periodic FDTI/BDTI trench benchmark for RCWA/FDTD alignment.

This is the next rung after the 1D slab/multilayer alignment ladder. It builds
one shared, periodic x-z trench geometry and evaluates it with:

1. torcwa RCWA using summed diffraction-order power.
2. A 2D TE Yee FDTD reference with periodic x and z-Poynting flux monitors.

The benchmark is intentionally smaller than a full CMOS image sensor pixel, but
it uses the same physical checks that matter before moving to a 2x2 Bayer pixel:
shared complex materials, shared geometry, R/T/A, grid/order convergence, and
visual field confinement around FDTI/BDTI trenches.
"""

from __future__ import annotations

import argparse
import json
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.materials.database import MaterialDB  # noqa: E402


@dataclass(frozen=True)
class Layer2D:
    name: str
    thickness_um: float
    eps_grid: np.ndarray


@dataclass(frozen=True)
class TrenchCase:
    mode: str
    wavelength_um: float
    pitch_um: float
    trench_width_um: float
    silicon_thickness_um: float
    bdti_depth_um: float
    y_period_um: float = 0.1


@dataclass
class Rta:
    reflection: float
    transmission: float
    absorption: float
    runtime_s: float

    def as_dict(self) -> dict[str, float]:
        return {
            "R": self.reflection,
            "T": self.transmission,
            "A": self.absorption,
            "R_plus_T_plus_A": self.reflection + self.transmission + self.absorption,
            "runtime_s": self.runtime_s,
        }


@dataclass
class Fdtd2DResult:
    rta: Rta
    x_um: np.ndarray
    z_um: np.ndarray
    eps_xz: np.ndarray
    intensity: np.ndarray
    silicon_absorption_proxy: float
    trench_field_leakage: float
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2D periodic FDTI/BDTI trench RCWA/FDTD benchmark.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "rcwa_fdtd_trench_benchmark",
        help="Output directory for plots and JSON report.",
    )
    parser.add_argument(
        "--dti-modes",
        choices=["fdti", "bdti"],
        nargs="+",
        default=["fdti", "bdti"],
        help="DTI modes to run.",
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=[0.50, 0.55, 0.60],
        help="Wavelengths in um.",
    )
    parser.add_argument("--pitch", type=float, default=1.0, help="Periodic pixel pitch in um.")
    parser.add_argument("--trench-width", type=float, default=0.10, help="DTI width in um.")
    parser.add_argument(
        "--silicon-thickness",
        type=float,
        default=1.20,
        help="Finite silicon benchmark thickness in um.",
    )
    parser.add_argument(
        "--bdti-depth",
        type=float,
        default=0.60,
        help="BDTI trench depth in um from the back/bottom side.",
    )
    parser.add_argument(
        "--rcwa-order",
        type=int,
        default=3,
        help="Primary RCWA x Fourier order. A small y order is used for numerical stability.",
    )
    parser.add_argument(
        "--rcwa-nx",
        type=int,
        default=160,
        help="RCWA permittivity samples per period in x.",
    )
    parser.add_argument(
        "--fdtd-dx",
        type=float,
        default=0.015,
        help="2D FDTD grid spacing in um.",
    )
    parser.add_argument(
        "--fdtd-runtime-um",
        type=float,
        default=45.0,
        help="2D FDTD runtime expressed as c*time in um.",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Run one-wavelength RCWA order and FDTD grid convergence sweeps.",
    )
    parser.add_argument(
        "--rcwa-orders",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="RCWA orders for convergence at the reference wavelength.",
    )
    parser.add_argument(
        "--fdtd-dx-values",
        type=float,
        nargs="+",
        default=[0.03, 0.02, 0.015],
        help="FDTD dx values for convergence at the reference wavelength.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one wavelength and coarser settings for smoke checks.",
    )
    return parser.parse_args()


def material_eps(db: MaterialDB, material: str, wavelength_um: float) -> complex:
    return complex(db.get_epsilon(material, wavelength_um))


def trench_mask_x(x_um: np.ndarray, pitch_um: float, trench_width_um: float) -> np.ndarray:
    half = trench_width_um / 2.0
    return (x_um < half) | (x_um >= pitch_um - half)


def patterned_silicon_eps(
    x_um: np.ndarray,
    silicon_eps: complex,
    trench_eps: complex,
    pitch_um: float,
    trench_width_um: float,
    ny: int,
) -> np.ndarray:
    row = np.full(x_um.size, silicon_eps, dtype=np.complex128)
    row[trench_mask_x(x_um, pitch_um, trench_width_um)] = trench_eps
    return np.repeat(row[np.newaxis, :], ny, axis=0)


def uniform_eps(eps: complex, nx: int, ny: int) -> np.ndarray:
    return np.full((ny, nx), eps, dtype=np.complex128)


def build_trench_layers(
    case: TrenchCase,
    nx: int,
    ny: int,
    db: MaterialDB,
) -> list[Layer2D]:
    """Build top-to-bottom RCWA/FDTD layers from one shared geometry model."""

    x_um = (np.arange(nx) + 0.5) * case.pitch_um / nx
    eps_air = material_eps(db, "air", case.wavelength_um)
    eps_polymer = material_eps(db, "polymer_n1p56", case.wavelength_um)
    eps_sio2 = material_eps(db, "sio2", case.wavelength_um)
    eps_cf = material_eps(db, "cf_green", case.wavelength_um)
    eps_hfo2 = material_eps(db, "hfo2", case.wavelength_um)
    eps_si3n4 = material_eps(db, "si3n4", case.wavelength_um)
    eps_si = material_eps(db, "silicon", case.wavelength_um)

    layers: list[Layer2D] = [
        Layer2D("microlens_equiv", 0.25, uniform_eps(eps_polymer, nx, ny)),
        Layer2D("planarization", 0.20, uniform_eps(eps_sio2, nx, ny)),
        Layer2D("green_cf", 0.35, uniform_eps(eps_cf, nx, ny)),
        Layer2D("barl_sio2", 0.035, uniform_eps(eps_sio2, nx, ny)),
        Layer2D("barl_hfo2", 0.045, uniform_eps(eps_hfo2, nx, ny)),
        Layer2D("barl_si3n4", 0.055, uniform_eps(eps_si3n4, nx, ny)),
    ]

    patterned_si = patterned_silicon_eps(
        x_um,
        silicon_eps=eps_si,
        trench_eps=eps_sio2,
        pitch_um=case.pitch_um,
        trench_width_um=case.trench_width_um,
        ny=ny,
    )
    uniform_si = uniform_eps(eps_si, nx, ny)

    if case.mode == "fdti":
        layers.append(Layer2D("silicon_fdti_full_depth", case.silicon_thickness_um, patterned_si))
    elif case.mode == "bdti":
        bdti_depth = float(np.clip(case.bdti_depth_um, 0.0, case.silicon_thickness_um))
        bulk_depth = case.silicon_thickness_um - bdti_depth
        if bulk_depth > 1e-9:
            layers.append(Layer2D("silicon_bulk_front_side", bulk_depth, uniform_si))
        if bdti_depth > 1e-9:
            layers.append(Layer2D("silicon_bdti_back_side", bdti_depth, patterned_si))
    else:
        layers.append(Layer2D("silicon_no_dti", case.silicon_thickness_um, uniform_si))

    # Keep a finite air spacer so both solvers have the same output medium.
    layers.append(Layer2D("exit_air", 0.20, uniform_eps(eps_air, nx, ny)))
    return layers


def run_torcwa_trench(
    case: TrenchCase,
    order: int,
    nx: int,
    db: MaterialDB,
) -> Rta:
    t0 = perf_counter()
    import torch
    import torcwa

    y_order = 1
    ny = 16
    layers = build_trench_layers(case, nx=nx, ny=ny, db=db)
    sim = torcwa.rcwa(
        freq=1.0 / case.wavelength_um,
        order=[order, y_order],
        L=[case.pitch_um, case.y_period_um],
        dtype=torch.complex64,
        device=torch.device("cpu"),
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(0.0, 0.0)
    for layer in layers:
        # Internal geometry is stored as (ny, nx), while torcwa's Fourier
        # convolution indexes the first tensor axis with the first lattice
        # vector/order. Transpose so order=[Nx, 0] sees the x-periodic trench.
        eps_tensor = torch.tensor(layer.eps_grid.T.copy(), dtype=torch.complex64)
        sim.add_layer(thickness=layer.thickness_um, eps=eps_tensor)
    sim.solve_global_smatrix()
    sim.source_planewave(amplitude=[0.0, 1.0], direction="forward", notation="xy")

    r = diffraction_power_sum(sim, torch, order, y_order, port="reflection")
    t = diffraction_power_sum(sim, torch, order, y_order, port="transmission")
    a = 1.0 - r - t
    return Rta(r, t, a, perf_counter() - t0)


def diffraction_power_sum(sim: Any, torch: Any, x_order: int, y_order: int, port: str) -> float:
    total = 0.0
    for mx in range(-x_order, x_order + 1):
        for my in range(-y_order, y_order + 1):
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
    return float(total)


def build_fdtd_eps_map(
    case: TrenchCase,
    dx_um: float,
    db: MaterialDB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    nx = max(24, round(case.pitch_um / dx_um))
    dx_eff = case.pitch_um / nx
    dz_um = dx_eff
    pml_cells = max(24, round(0.60 / dz_um))
    top_air_um = 1.0
    bottom_air_um = 1.0

    layers = build_trench_layers(case, nx=nx, ny=1, db=db)
    stack_um = sum(layer.thickness_um for layer in layers)
    total_um = top_air_um + stack_um + bottom_air_um + 2 * pml_cells * dz_um
    nz = int(np.ceil(total_um / dz_um))

    x_um = (np.arange(nx) + 0.5) * dx_eff
    z_um = (np.arange(nz) + 0.5) * dz_um - pml_cells * dz_um - top_air_um
    eps_air = material_eps(db, "air", case.wavelength_um)
    eps = np.full((nz, nx), eps_air, dtype=np.complex128)

    layer_bounds = []
    z_cursor = 0.0
    for layer in layers:
        z_next = z_cursor + layer.thickness_um
        z_mask = (z_um >= z_cursor) & (z_um < z_next)
        if np.any(z_mask):
            eps[z_mask, :] = layer.eps_grid[0, :][np.newaxis, :]
        layer_bounds.append(
            {
                "name": layer.name,
                "z_start_um": z_cursor,
                "z_end_um": z_next,
                "thickness_um": layer.thickness_um,
            }
        )
        z_cursor = z_next

    metadata = {
        "nx": nx,
        "nz": nz,
        "dx_um": dx_eff,
        "dz_um": dz_um,
        "pml_cells": pml_cells,
        "top_air_um": top_air_um,
        "bottom_air_um": bottom_air_um,
        "stack_thickness_um": stack_um,
        "layer_bounds": layer_bounds,
    }
    return x_um, z_um, eps, metadata


def run_fdtd_2d_te(
    case: TrenchCase,
    dx_um: float,
    runtime_um: float,
    db: MaterialDB,
    courant: float = 0.45,
) -> Fdtd2DResult:
    t0 = perf_counter()
    x_um, z_um, eps_structure, metadata = build_fdtd_eps_map(case, dx_um, db)
    eps_reference = np.ones_like(eps_structure)
    nz, nx = eps_structure.shape
    dx_eff = metadata["dx_um"]
    dz_eff = metadata["dz_um"]
    pml_cells = metadata["pml_cells"]

    source_z = pml_cells + max(4, round(0.22 / dz_eff))
    reflect_z = pml_cells + max(8, round(0.58 / dz_eff))
    stack_start_z = pml_cells + round(metadata["top_air_um"] / dz_eff)
    stack_end_z = stack_start_z + round(metadata["stack_thickness_um"] / dz_eff)
    transmit_z = min(nz - pml_cells - 5, stack_end_z + max(8, round(0.55 / dz_eff)))
    if not (source_z < reflect_z < stack_start_z):
        raise RuntimeError("Invalid FDTD source/reflection monitor placement")
    if not (stack_end_z < transmit_z < nz - pml_cells):
        raise RuntimeError("Invalid FDTD transmission monitor placement")

    dt = courant / np.sqrt((1.0 / dx_eff**2) + (1.0 / dz_eff**2))
    steps = max(500, round(runtime_um / dt))
    warmup = int(0.55 * steps)
    ramp_steps = max(30, int(0.18 * steps))
    omega = 2.0 * np.pi / case.wavelength_um
    damp_e = fdtd_z_damping(nz, pml_cells)[:, np.newaxis]
    damp_hx = 0.5 * (damp_e[:-1] + damp_e[1:])

    def run(eps_complex: np.ndarray, capture_field: bool) -> dict[str, Any]:
        eps_real = np.maximum(np.real(eps_complex), 1e-9)
        sigma = np.maximum(np.imag(eps_complex), 0.0) * omega
        alpha = sigma * dt / (2.0 * eps_real)
        e_decay = (1.0 - alpha) / (1.0 + alpha)
        e_curl = dt / eps_real / (1.0 + alpha)

        ey = np.zeros((nz, nx), dtype=np.float64)
        hx = np.zeros((nz - 1, nx), dtype=np.float64)
        hz = np.zeros((nz, nx), dtype=np.float64)
        e_reflect = np.zeros(nx, dtype=np.complex128)
        h_reflect = np.zeros(nx, dtype=np.complex128)
        e_transmit = np.zeros(nx, dtype=np.complex128)
        h_transmit = np.zeros(nx, dtype=np.complex128)
        field_accum = np.zeros_like(eps_complex, dtype=np.complex128) if capture_field else None
        count = 0

        for step in range(steps):
            hx += (dt / dz_eff) * (ey[1:, :] - ey[:-1, :])
            hz -= (dt / dx_eff) * (np.roll(ey, -1, axis=1) - ey)
            hx *= damp_hx
            hz *= damp_e

            curl = np.zeros_like(ey)
            curl[1:-1, :] = (
                (hx[1:, :] - hx[:-1, :]) / dz_eff
                - (hz[1:-1, :] - np.roll(hz[1:-1, :], 1, axis=1)) / dx_eff
            )
            ey[1:-1, :] = e_decay[1:-1, :] * ey[1:-1, :] + e_curl[1:-1, :] * curl[1:-1, :]

            ramp = 1.0
            if step < ramp_steps:
                ramp = np.sin(0.5 * np.pi * step / ramp_steps) ** 2
            ey[source_z, :] += 0.25 * ramp * np.sin(omega * step * dt)
            ey *= damp_e

            if step >= warmup:
                phase = np.exp(-1j * omega * step * dt)
                e_reflect += ey[reflect_z, :] * phase
                h_reflect += hx[reflect_z, :] * phase
                e_transmit += ey[transmit_z, :] * phase
                h_transmit += hx[transmit_z, :] * phase
                if field_accum is not None:
                    field_accum += ey * phase
                count += 1

        scale = 2.0 / max(count, 1)
        result: dict[str, Any] = {
            "e_reflect": scale * e_reflect,
            "h_reflect": scale * h_reflect,
            "e_transmit": scale * e_transmit,
            "h_transmit": scale * h_transmit,
        }
        if field_accum is not None:
            result["field"] = scale * field_accum
        return result

    ref = run(eps_reference, capture_field=False)
    sim = run(eps_structure, capture_field=True)

    raw_reflect_flux = poynting_flux(ref["e_reflect"], ref["h_reflect"])
    flux_sign = 1.0 if raw_reflect_flux >= 0.0 else -1.0
    incident_flux = max(flux_sign * raw_reflect_flux, 1e-30)
    ref_transmit_flux = max(flux_sign * poynting_flux(ref["e_transmit"], ref["h_transmit"]), 1e-30)
    structure_reflect_plane_flux = flux_sign * poynting_flux(sim["e_reflect"], sim["h_reflect"])
    structure_transmit_flux = flux_sign * poynting_flux(sim["e_transmit"], sim["h_transmit"])

    reflection = float(np.clip((incident_flux - structure_reflect_plane_flux) / incident_flux, 0.0, 2.0))
    transmission = float(np.clip(structure_transmit_flux / ref_transmit_flux, 0.0, 2.0))
    absorption = float(1.0 - reflection - transmission)

    field = sim["field"]
    intensity = np.abs(field) ** 2
    intensity /= max(float(np.nanmax(intensity)), 1e-30)
    silicon_mask = silicon_region_mask(z_um, eps_structure, db, case.wavelength_um)
    trench_mask = trench_region_mask(x_um, z_um, case)
    sigma = np.maximum(np.imag(eps_structure), 0.0) * omega
    loss_density = sigma * intensity
    silicon_absorption_proxy = float(
        np.sum(loss_density[silicon_mask]) / max(float(np.sum(loss_density)), 1e-30)
    )
    trench_field = float(np.mean(intensity[trench_mask])) if np.any(trench_mask) else 0.0
    silicon_field = float(np.mean(intensity[silicon_mask & ~trench_mask])) if np.any(silicon_mask & ~trench_mask) else 1.0
    trench_field_leakage = trench_field / max(silicon_field, 1e-30)

    metadata.update(
        {
            "steps": steps,
            "dt_um_over_c": dt,
            "source_z": source_z,
            "reflect_z": reflect_z,
            "transmit_z": transmit_z,
            "incident_flux": incident_flux,
            "reference_transmit_flux": ref_transmit_flux,
            "runtime_seconds": perf_counter() - t0,
        }
    )
    return Fdtd2DResult(
        rta=Rta(reflection, transmission, absorption, perf_counter() - t0),
        x_um=x_um,
        z_um=z_um,
        eps_xz=eps_structure,
        intensity=intensity,
        silicon_absorption_proxy=silicon_absorption_proxy,
        trench_field_leakage=trench_field_leakage,
        metadata=metadata,
    )


def poynting_flux(e_y: np.ndarray, h_x: np.ndarray) -> float:
    return float(0.5 * np.real(np.mean(e_y * np.conj(h_x))))


def fdtd_z_damping(nz: int, pml_cells: int) -> np.ndarray:
    damping = np.ones(nz, dtype=np.float64)
    for i in range(pml_cells):
        strength = ((pml_cells - i) / pml_cells) ** 3
        value = np.exp(-0.10 * strength)
        damping[i] = min(damping[i], value)
        damping[-1 - i] = min(damping[-1 - i], value)
    return damping


def silicon_region_mask(
    z_um: np.ndarray,
    eps_xz: np.ndarray,
    db: MaterialDB,
    wavelength_um: float,
) -> np.ndarray:
    eps_si = material_eps(db, "silicon", wavelength_um)
    return np.abs(eps_xz - eps_si) < np.abs(eps_si) * 0.35


def trench_region_mask(x_um: np.ndarray, z_um: np.ndarray, case: TrenchCase) -> np.ndarray:
    x_mask = trench_mask_x(x_um, case.pitch_um, case.trench_width_um)
    silicon_top = 0.25 + 0.20 + 0.35 + 0.035 + 0.045 + 0.055
    silicon_bottom = silicon_top + case.silicon_thickness_um
    if case.mode == "fdti":
        z_mask = (z_um >= silicon_top) & (z_um < silicon_bottom)
    else:
        z_start = silicon_bottom - min(case.bdti_depth_um, case.silicon_thickness_um)
        z_mask = (z_um >= z_start) & (z_um < silicon_bottom)
    return z_mask[:, np.newaxis] & x_mask[np.newaxis, :]


def run_spectrum(args: argparse.Namespace, db: MaterialDB) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode in args.dti_modes:
        rows = []
        fdtd_by_wavelength = {}
        for wavelength in args.wavelengths:
            case = TrenchCase(
                mode=mode,
                wavelength_um=float(wavelength),
                pitch_um=args.pitch,
                trench_width_um=args.trench_width,
                silicon_thickness_um=args.silicon_thickness,
                bdti_depth_um=args.bdti_depth,
            )
            rcwa, rcwa_order_used, rcwa_errors = run_torcwa_with_fallback(
                case,
                preferred_order=args.rcwa_order,
                nx=args.rcwa_nx,
                db=db,
            )
            fdtd = run_fdtd_2d_te(case, dx_um=args.fdtd_dx, runtime_um=args.fdtd_runtime_um, db=db)
            fdtd_by_wavelength[str(wavelength)] = fdtd
            rows.append(
                {
                    "wavelength_um": float(wavelength),
                    "rcwa_order_used": rcwa_order_used,
                    "rcwa_fallback_errors": rcwa_errors,
                    "rcwa_torcwa": rcwa.as_dict(),
                    "fdtd_2d_te": fdtd.rta.as_dict(),
                    "abs_error": {
                        "R": abs(rcwa.reflection - fdtd.rta.reflection),
                        "T": abs(rcwa.transmission - fdtd.rta.transmission),
                        "A": abs(rcwa.absorption - fdtd.rta.absorption),
                    },
                    "fdtd_silicon_absorption_proxy": fdtd.silicon_absorption_proxy,
                    "fdtd_trench_field_leakage": fdtd.trench_field_leakage,
                    "fdtd_metadata": fdtd.metadata,
                }
            )
        results[mode] = {"spectrum": rows, "fdtd_objects": fdtd_by_wavelength}
    return results


def run_torcwa_with_fallback(
    case: TrenchCase,
    preferred_order: int,
    nx: int,
    db: MaterialDB,
) -> tuple[Rta, int, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    for order in range(preferred_order, 0, -1):
        try:
            return run_torcwa_trench(case, order=order, nx=nx, db=db), order, errors
        except Exception as exc:
            errors.append({"order": str(order), "error": repr(exc)})
    raise RuntimeError(
        f"torcwa failed for all fallback orders at {case.wavelength_um} um, mode={case.mode}: "
        f"{errors}"
    )


def run_convergence(args: argparse.Namespace, db: MaterialDB) -> dict[str, Any]:
    if not args.convergence:
        return {}
    reference_wavelength = 0.55
    convergence: dict[str, Any] = {}
    for mode in args.dti_modes:
        case = TrenchCase(
            mode=mode,
            wavelength_um=reference_wavelength,
            pitch_um=args.pitch,
            trench_width_um=args.trench_width,
            silicon_thickness_um=args.silicon_thickness,
            bdti_depth_um=args.bdti_depth,
        )
        rcwa_rows = []
        for order in args.rcwa_orders:
            try:
                result = run_torcwa_trench(case, order=order, nx=args.rcwa_nx, db=db)
                rcwa_rows.append({"order": order, "ok": True, **result.as_dict()})
            except Exception as exc:
                rcwa_rows.append({"order": order, "ok": False, "error": repr(exc)})
        fdtd_rows = []
        for dx in args.fdtd_dx_values:
            result = run_fdtd_2d_te(case, dx_um=dx, runtime_um=args.fdtd_runtime_um, db=db)
            fdtd_rows.append({"dx_um": dx, **result.rta.as_dict()})
        convergence[mode] = {
            "wavelength_um": reference_wavelength,
            "rcwa_order_sweep": rcwa_rows,
            "fdtd_dx_sweep": fdtd_rows,
        }
    return convergence


def summarize(results: dict[str, Any], convergence: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode, payload in results.items():
        rows = payload["spectrum"]
        max_errors = {
            key: max(row["abs_error"][key] for row in rows)
            for key in ("R", "T", "A")
        }
        summary[mode] = {
            "max_abs_error_rcwa_vs_fdtd": max_errors,
            "mean_fdtd_silicon_absorption_proxy": float(
                np.mean([row["fdtd_silicon_absorption_proxy"] for row in rows])
            ),
            "mean_fdtd_trench_field_leakage": float(
                np.mean([row["fdtd_trench_field_leakage"] for row in rows])
            ),
            "status": {
                "rta_aligned_for_periodic_trench": (
                    max_errors["R"] < 0.08
                    and max_errors["T"] < 0.08
                    and max_errors["A"] < 0.12
                )
            },
        }
        if mode in convergence:
            summary[mode]["convergence_reference_wavelength_um"] = convergence[mode]["wavelength_um"]
    return summary


def plot_geometry(args: argparse.Namespace, db: MaterialDB, outpath: Path) -> None:
    n_modes = len(args.dti_modes)
    fig, axes = plt.subplots(1, n_modes, figsize=(6.2 * n_modes, 4.8), constrained_layout=True)
    if n_modes == 1:
        axes = [axes]
    for ax, mode in zip(axes, args.dti_modes):
        case = TrenchCase(
            mode=mode,
            wavelength_um=0.55,
            pitch_um=args.pitch,
            trench_width_um=args.trench_width,
            silicon_thickness_um=args.silicon_thickness,
            bdti_depth_um=args.bdti_depth,
        )
        x_um, z_um, eps, _metadata = build_fdtd_eps_map(case, args.fdtd_dx, db)
        im = ax.imshow(
            np.real(eps),
            extent=[x_um[0], x_um[-1], z_um[-1], z_um[0]],
            aspect="auto",
            cmap="viridis",
        )
        ax.set_title(f"{mode.upper()} shared geometry, Re(eps)")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("depth from optical stack top (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_spectrum(results: dict[str, Any], wavelengths: list[float], outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    colors = {"fdti": "#7c3aed", "bdti": "#0f766e"}
    for mode, payload in results.items():
        rows = payload["spectrum"]
        wl_nm = np.array([row["wavelength_um"] for row in rows]) * 1000.0
        for ax, key, title in zip(axes, ["R", "T", "A"], ["Reflection", "Transmission", "Absorption"]):
            ax.plot(
                wl_nm,
                [row["rcwa_torcwa"][key] for row in rows],
                color=colors.get(mode, "black"),
                linewidth=2,
                label=f"{mode} RCWA" if key == "R" else None,
            )
            ax.plot(
                wl_nm,
                [row["fdtd_2d_te"][key] for row in rows],
                color=colors.get(mode, "black"),
                marker="o",
                linestyle="--",
                label=f"{mode} FDTD" if key == "R" else None,
            )
            ax.set_title(title)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("2D periodic FDTI/BDTI trench RCWA-FDTD alignment")
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_errors(results: dict[str, Any], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    colors = {"fdti": "#7c3aed", "bdti": "#0f766e"}
    for mode, payload in results.items():
        rows = payload["spectrum"]
        wl_nm = np.array([row["wavelength_um"] for row in rows]) * 1000.0
        for key, style in [("R", "-"), ("T", "--"), ("A", ":")]:
            err = np.maximum(np.array([row["abs_error"][key] for row in rows]), 1e-16)
            ax.semilogy(
                wl_nm,
                err,
                color=colors.get(mode, "black"),
                linestyle=style,
                marker="o",
                label=f"{mode} |d{key}|",
            )
    ax.set_title("Absolute RCWA-FDTD R/T/A difference")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("absolute error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_fields(results: dict[str, Any], reference_wavelength: float, outpath: Path) -> None:
    n_modes = len(results)
    fig, axes = plt.subplots(1, n_modes, figsize=(6.2 * n_modes, 4.8), constrained_layout=True)
    if n_modes == 1:
        axes = [axes]
    for ax, (mode, payload) in zip(axes, results.items()):
        fdtd = payload["fdtd_objects"][str(reference_wavelength)]
        intensity = np.maximum(fdtd.intensity, 1e-6)
        im = ax.imshow(
            intensity,
            extent=[fdtd.x_um[0], fdtd.x_um[-1], fdtd.z_um[-1], fdtd.z_um[0]],
            aspect="auto",
            norm=LogNorm(vmin=1e-5, vmax=1.0),
            cmap="magma",
        )
        ax.contour(
            fdtd.x_um,
            fdtd.z_um,
            np.real(fdtd.eps_xz),
            levels=5,
            colors="white",
            linewidths=0.35,
            alpha=0.5,
        )
        ax.set_title(f"{mode.upper()} FDTD |Ey|^2 at {reference_wavelength * 1000:.0f} nm")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("depth from optical stack top (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_convergence(convergence: dict[str, Any], outpath: Path) -> None:
    if not convergence:
        return
    n_modes = len(convergence)
    fig, axes = plt.subplots(n_modes, 2, figsize=(11, 4.2 * n_modes), constrained_layout=True)
    if n_modes == 1:
        axes = np.asarray([axes])
    for row_axes, (mode, payload) in zip(axes, convergence.items()):
        rcwa_rows = payload["rcwa_order_sweep"]
        fdtd_rows = payload["fdtd_dx_sweep"]
        rcwa_ok = [row for row in rcwa_rows if row.get("ok", True)]
        orders = [row["order"] for row in rcwa_ok]
        dxs = [row["dx_um"] for row in fdtd_rows]
        for key, style in [("R", "-"), ("T", "--"), ("A", ":")]:
            row_axes[0].plot(orders, [row[key] for row in rcwa_ok], marker="o", linestyle=style, label=key)
            row_axes[1].plot(dxs, [row[key] for row in fdtd_rows], marker="o", linestyle=style, label=key)
        row_axes[0].set_title(f"{mode} RCWA order convergence")
        row_axes[0].set_xlabel("Fourier order")
        row_axes[0].set_ylabel("R/T/A")
        row_axes[0].grid(True, alpha=0.3)
        row_axes[0].legend(fontsize=8)
        row_axes[1].invert_xaxis()
        row_axes[1].set_title(f"{mode} FDTD dx convergence")
        row_axes[1].set_xlabel("dx (um)")
        row_axes[1].set_ylabel("R/T/A")
        row_axes[1].grid(True, alpha=0.3)
        row_axes[1].legend(fontsize=8)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def write_report(
    outdir: Path,
    args: argparse.Namespace,
    summary: dict[str, Any],
    plots: dict[str, str],
) -> None:
    lines = [
        "# 2D Periodic Trench RCWA/FDTD Benchmark",
        "",
        f"- DTI modes: `{', '.join(args.dti_modes)}`",
        f"- Pitch: `{args.pitch}` um",
        f"- Trench width: `{args.trench_width}` um",
        f"- Silicon thickness: `{args.silicon_thickness}` um",
        f"- BDTI depth: `{args.bdti_depth}` um",
        f"- RCWA order: `{args.rcwa_order}`",
        f"- FDTD dx: `{args.fdtd_dx}` um",
        f"- FDTD runtime c*t: `{args.fdtd_runtime_um}` um",
        "",
        "The RCWA result sums available x diffraction orders. The FDTD result",
        "uses periodic-x TE Yee fields and z-directed Poynting flux monitors.",
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
    (outdir / "trench_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def serializable_results(results: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for mode, payload in results.items():
        clean[mode] = {"spectrum": payload["spectrum"]}
    return clean


def main() -> None:
    args = parse_args()
    if args.quick:
        args.wavelengths = [0.55]
        args.rcwa_order = min(args.rcwa_order, 3)
        args.rcwa_nx = min(args.rcwa_nx, 96)
        args.fdtd_dx = max(args.fdtd_dx, 0.025)
        args.fdtd_runtime_um = min(args.fdtd_runtime_um, 28.0)
        args.convergence = False

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    db = MaterialDB()

    results = run_spectrum(args, db)
    convergence = run_convergence(args, db)
    summary = summarize(results, convergence)

    metrics = {
        "settings": {
            "dti_modes": args.dti_modes,
            "wavelengths_um": args.wavelengths,
            "pitch_um": args.pitch,
            "trench_width_um": args.trench_width,
            "silicon_thickness_um": args.silicon_thickness,
            "bdti_depth_um": args.bdti_depth,
            "rcwa_order": args.rcwa_order,
            "rcwa_nx": args.rcwa_nx,
            "fdtd_dx_um": args.fdtd_dx,
            "fdtd_runtime_um": args.fdtd_runtime_um,
        },
        "summary": summary,
        "results": serializable_results(results),
        "convergence": convergence,
    }
    (outdir / "trench_benchmark_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    reference_wavelength = min(args.wavelengths, key=lambda wl: abs(wl - 0.55))
    plots = {
        "Shared Geometry": "01_shared_geometry.png",
        "Spectrum Alignment": "02_spectrum_alignment.png",
        "RTA Error": "03_rta_error.png",
        "FDTD Field Maps": "04_fdtd_field_maps.png",
    }
    if convergence:
        plots["Convergence"] = "05_convergence.png"

    plot_geometry(args, db, outdir / plots["Shared Geometry"])
    plot_spectrum(results, args.wavelengths, outdir / plots["Spectrum Alignment"])
    plot_errors(results, outdir / plots["RTA Error"])
    plot_fields(results, reference_wavelength, outdir / plots["FDTD Field Maps"])
    if convergence:
        plot_convergence(convergence, outdir / plots["Convergence"])
    write_report(outdir, args, summary, plots)

    print(f"2D trench benchmark artifacts written to: {outdir}")
    print(f"  - {outdir / 'trench_benchmark_metrics.json'}")
    print(f"  - {outdir / 'trench_benchmark_report.md'}")
    for filename in plots.values():
        print(f"  - {outdir / filename}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
