#!/usr/bin/env python3
"""Align RCWA and FDTD against shared 1D optical reference problems.

This is intentionally separate from the CMOS pixel visual proxy. The purpose
is to remove geometry/source ambiguity and prove that the solver chain agrees
on problems with an analytical transfer-matrix reference:

    air / lossless dielectric slab / air
    air / lossless pixel-like multilayer / air

The ladder here is:
1. TMM reference.
2. torcwa RCWA at zero Fourier order for the same uniform stack.
3. 1D Yee FDTD with DFT lock-in monitors and forward/backward wave separation.

Once this passes, the same source, monitor, normalization, and convergence
discipline should be carried upward to trenches and finally the full CMOS pixel.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.solvers.tmm.tmm_core import transfer_matrix_1d  # noqa: E402


@dataclass(frozen=True)
class LayerSpec:
    name: str
    refractive_index: complex
    thickness_um: float


@dataclass(frozen=True)
class AlignmentCase:
    structure_name: str
    wavelength_um: float
    layers: tuple[LayerSpec, ...]
    incident_index: float = 1.0
    substrate_index: float = 1.0

    @property
    def stack_thickness_um(self) -> float:
        return sum(layer.thickness_um for layer in self.layers)

    @property
    def layer_summary(self) -> list[dict[str, float | str]]:
        return [
            {
                "name": layer.name,
                "n_real": float(np.real(layer.refractive_index)),
                "n_imag": float(np.imag(layer.refractive_index)),
                "thickness_um": layer.thickness_um,
            }
            for layer in self.layers
        ]


@dataclass(frozen=True)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real RCWA/FDTD alignment ladder on a 1D dielectric stack.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "rcwa_fdtd_alignment",
        help="Output directory for plots and JSON report.",
    )
    parser.add_argument(
        "--structure",
        choices=["slab", "multilayer", "lossy-multilayer"],
        default="slab",
        help=(
            "1D reference structure to align. Use multilayer/lossy-multilayer "
            "for higher ladder rungs."
        ),
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=[0.45, 0.50, 0.55, 0.60, 0.65],
        help="Wavelengths in um.",
    )
    parser.add_argument(
        "--slab-index",
        type=float,
        default=1.5,
        help="Lossless slab refractive index for --structure slab.",
    )
    parser.add_argument(
        "--slab-thickness",
        type=float,
        default=0.2,
        help="Slab thickness in um for --structure slab.",
    )
    parser.add_argument(
        "--fdtd-dx",
        type=float,
        default=0.0025,
        help="FDTD grid spacing in um.",
    )
    parser.add_argument(
        "--fdtd-runtime-um",
        type=float,
        default=80.0,
        help="FDTD runtime expressed as c*time in um. Steps are runtime/(courant*dx).",
    )
    parser.add_argument(
        "--convergence-dx",
        type=float,
        nargs="+",
        default=[0.02, 0.01, 0.005, 0.0025],
        help="FDTD dx values for single-wavelength convergence sweep.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer wavelengths and a shorter convergence sweep.",
    )
    return parser.parse_args()


def build_alignment_case(
    structure: str,
    wavelength_um: float,
    slab_index: float,
    slab_thickness_um: float,
) -> AlignmentCase:
    if structure == "slab":
        return AlignmentCase(
            structure_name="lossless_slab",
            wavelength_um=wavelength_um,
            layers=(LayerSpec("slab", slab_index, slab_thickness_um),),
        )
    if structure == "multilayer":
        return AlignmentCase(
            structure_name="lossless_pixel_like_multilayer",
            wavelength_um=wavelength_um,
            layers=(
                LayerSpec("polymer_lens_equiv", 1.56, 0.22),
                LayerSpec("oxide_planarization", 1.46, 0.18),
                LayerSpec("green_cf_equiv", 1.72, 0.26),
                LayerSpec("barl_sio2", 1.46, 0.04),
                LayerSpec("barl_hfo2", 2.05, 0.05),
                LayerSpec("barl_si3n4", 2.00, 0.06),
                LayerSpec("silicon_proxy", 3.50, 0.18),
            ),
        )
    if structure == "lossy-multilayer":
        return AlignmentCase(
            structure_name="lossy_pixel_like_multilayer",
            wavelength_um=wavelength_um,
            layers=(
                LayerSpec("polymer_lens_equiv", 1.56 + 0.0j, 0.22),
                LayerSpec("oxide_planarization", 1.46 + 0.0j, 0.18),
                LayerSpec("green_cf_absorber", 1.72 + 0.025j, 0.26),
                LayerSpec("barl_sio2", 1.46 + 0.0j, 0.04),
                LayerSpec("barl_hfo2", 2.05 + 0.0j, 0.05),
                LayerSpec("barl_si3n4", 2.00 + 0.0j, 0.06),
                LayerSpec("silicon_absorber", 3.50 + 0.050j, 0.18),
            ),
        )
    raise ValueError(f"Unknown structure '{structure}'")


def tmm_stack(case: AlignmentCase) -> Rta:
    t0 = perf_counter()
    r, t, a = transfer_matrix_1d(
        n_layers=np.array(
            [case.incident_index]
            + [layer.refractive_index for layer in case.layers]
            + [case.substrate_index],
            dtype=complex,
        ),
        d_layers=np.array(
            [np.inf]
            + [layer.thickness_um for layer in case.layers]
            + [np.inf],
            dtype=float,
        ),
        wavelength=case.wavelength_um,
        theta_inc=0.0,
        polarization="TE",
    )
    return Rta(r, t, a, perf_counter() - t0)


def torcwa_stack(case: AlignmentCase) -> Rta:
    t0 = perf_counter()
    import torch
    import torcwa

    sim = torcwa.rcwa(
        freq=1.0 / case.wavelength_um,
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


def fdtd_1d_stack(
    case: AlignmentCase,
    dx_um: float,
    runtime_um: float,
    courant: float = 0.5,
) -> Rta:
    """Run a 1D Yee FDTD stack solve with DFT monitor normalization.

    The update is in normalized units where c=1 and distances are in um.
    The reflection monitor separates forward/backward waves with:

        E_forward = (E - H) / 2
        E_backward = (E + H) / 2

    for the sign convention used by the update equations. The incident and
    substrate media are air in the current alignment cases, so no admittance
    correction is needed for the normalized transmitted power.
    """

    t0 = perf_counter()
    pml_cells = max(80, round(2.0 / dx_um))
    air_left_um = 2.0
    air_right_um = 2.0
    domain_um = air_left_um + case.stack_thickness_um + air_right_um + 2 * pml_cells * dx_um
    nx = round(domain_um / dx_um)
    z = np.arange(nx) * dx_um

    stack_start = pml_cells * dx_um + air_left_um
    stack_end = stack_start + case.stack_thickness_um
    eps_structure = np.full(nx, case.incident_index**2, dtype=np.complex128)
    eps_structure[z >= stack_end] = case.substrate_index**2
    current_z = stack_start
    for layer in case.layers:
        next_z = current_z + layer.thickness_um
        eps_structure[(z >= current_z) & (z < next_z)] = layer.refractive_index**2
        current_z = next_z
    eps_reference = np.ones_like(eps_structure)

    source_idx = round((pml_cells * dx_um + 0.55) / dx_um)
    reflect_idx = round((pml_cells * dx_um + 1.15) / dx_um)
    transmit_idx = round((stack_end + 0.70) / dx_um)
    if not (source_idx < reflect_idx < round(stack_start / dx_um)):
        raise RuntimeError("Invalid FDTD source/monitor placement")
    if not (round(stack_end / dx_um) < transmit_idx < nx - pml_cells):
        raise RuntimeError("Invalid FDTD transmission monitor placement")

    dt = courant * dx_um
    steps = max(200, round(runtime_um / dt))
    omega = 2.0 * np.pi / case.wavelength_um
    damping_e = fdtd_damping(nx, pml_cells)
    damping_h = 0.5 * (damping_e[:-1] + damping_e[1:])

    def run(eps: np.ndarray) -> tuple[complex, complex, complex, complex]:
        eps_real = np.maximum(np.real(eps), 1e-9)
        sigma = np.maximum(np.imag(eps), 0.0) * omega
        alpha = sigma * dt / (2.0 * eps_real)
        e_decay = (1.0 - alpha) / (1.0 + alpha)
        e_curl = (dt / dx_um) / eps_real / (1.0 + alpha)
        e_field = np.zeros(nx, dtype=np.float64)
        h_field = np.zeros(nx - 1, dtype=np.float64)
        e_reflect = 0j
        h_reflect = 0j
        e_transmit = 0j
        h_transmit = 0j
        count = 0
        warmup = int(0.55 * steps)
        ramp_steps = max(20, int(0.15 * steps))

        for step in range(steps):
            h_field += (dt / dx_um) * (e_field[1:] - e_field[:-1])
            h_field *= damping_h
            e_field[1:-1] = e_decay[1:-1] * e_field[1:-1] + e_curl[1:-1] * (
                h_field[1:] - h_field[:-1]
            )

            ramp = 1.0
            if step < ramp_steps:
                ramp = np.sin(0.5 * np.pi * step / ramp_steps) ** 2
            e_field[source_idx] += ramp * np.sin(omega * step * dt)
            e_field *= damping_e

            if step >= warmup:
                phase = np.exp(-1j * omega * step * dt)
                e_reflect += e_field[reflect_idx] * phase
                h_reflect += h_field[reflect_idx] * phase
                e_transmit += e_field[transmit_idx] * phase
                h_transmit += h_field[transmit_idx] * phase
                count += 1

        scale = 2.0 / max(count, 1)
        return (
            scale * e_reflect,
            scale * h_reflect,
            scale * e_transmit,
            scale * h_transmit,
        )

    ref_er, ref_hr, ref_et, ref_ht = run(eps_reference)
    stack_er, stack_hr, stack_et, stack_ht = run(eps_structure)

    incident_left = (ref_er - ref_hr) / 2.0
    reference_back_left = (ref_er + ref_hr) / 2.0
    reflected_left = (stack_er + stack_hr) / 2.0 - reference_back_left
    incident_transmit = (ref_et - ref_ht) / 2.0
    transmitted = (stack_et - stack_ht) / 2.0

    r = float(abs(reflected_left) ** 2 / max(abs(incident_left) ** 2, 1e-30))
    t = float(abs(transmitted) ** 2 / max(abs(incident_transmit) ** 2, 1e-30))
    a = 1.0 - r - t
    return Rta(r, t, a, perf_counter() - t0)


def fdtd_damping(nx: int, pml_cells: int) -> np.ndarray:
    damping = np.ones(nx, dtype=np.float64)
    for i in range(pml_cells):
        strength = ((pml_cells - i) / pml_cells) ** 3
        value = np.exp(-0.08 * strength)
        damping[i] = min(damping[i], value)
        damping[-1 - i] = min(damping[-1 - i], value)
    return damping


def run_spectrum(
    structure: str,
    wavelengths: np.ndarray,
    slab_index: float,
    slab_thickness_um: float,
    fdtd_dx_um: float,
    fdtd_runtime_um: float,
) -> dict[str, list[dict[str, float]]]:
    results = {"tmm": [], "rcwa_torcwa": [], "fdtd_1d": []}
    for wavelength in wavelengths:
        case = build_alignment_case(
            structure,
            wavelength_um=float(wavelength),
            slab_index=slab_index,
            slab_thickness_um=slab_thickness_um,
        )
        results["tmm"].append(tmm_stack(case).as_dict())
        results["rcwa_torcwa"].append(torcwa_stack(case).as_dict())
        results["fdtd_1d"].append(
            fdtd_1d_stack(case, dx_um=fdtd_dx_um, runtime_um=fdtd_runtime_um).as_dict()
        )
    return results


def run_fdtd_convergence(
    structure: str,
    wavelength: float,
    slab_index: float,
    slab_thickness_um: float,
    dx_values: list[float],
    runtime_um: float,
) -> list[dict[str, float]]:
    case = build_alignment_case(structure, wavelength, slab_index, slab_thickness_um)
    reference = tmm_stack(case)
    rows = []
    for dx in dx_values:
        result = fdtd_1d_stack(case, dx_um=dx, runtime_um=runtime_um)
        rows.append(
            {
                "dx_um": dx,
                **result.as_dict(),
                "abs_R_error": abs(result.reflection - reference.reflection),
                "abs_T_error": abs(result.transmission - reference.transmission),
                "energy_residual": abs(
                    result.reflection + result.transmission + result.absorption - 1.0
                ),
            }
        )
    return rows


def summarize_alignment(
    case: AlignmentCase,
    wavelengths: np.ndarray,
    spectrum: dict[str, list[dict[str, float]]],
    convergence: list[dict[str, float]],
) -> dict[str, Any]:
    tmm = spectrum["tmm"]
    rcwa = spectrum["rcwa_torcwa"]
    fdtd = spectrum["fdtd_1d"]

    def max_abs_error(method: list[dict[str, float]], key: str) -> float:
        return max(abs(row[key] - ref[key]) for row, ref in zip(method, tmm))

    summary = {
        "structure_name": case.structure_name,
        "wavelengths_um": wavelengths.tolist(),
        "max_abs_error_vs_tmm": {
            "rcwa_torcwa_R": max_abs_error(rcwa, "R"),
            "rcwa_torcwa_T": max_abs_error(rcwa, "T"),
            "fdtd_1d_R": max_abs_error(fdtd, "R"),
            "fdtd_1d_T": max_abs_error(fdtd, "T"),
        },
        "fdtd_finest_dx_um": convergence[-1]["dx_um"] if convergence else None,
        "fdtd_finest_abs_R_error": convergence[-1]["abs_R_error"] if convergence else None,
        "fdtd_finest_abs_T_error": convergence[-1]["abs_T_error"] if convergence else None,
    }
    summary["status"] = {
        "rcwa_torcwa_aligned_to_tmm": (
            summary["max_abs_error_vs_tmm"]["rcwa_torcwa_R"] < 1e-5
            and summary["max_abs_error_vs_tmm"]["rcwa_torcwa_T"] < 1e-5
        ),
        "fdtd_1d_aligned_to_tmm": (
            summary["max_abs_error_vs_tmm"]["fdtd_1d_R"] < 5e-3
            and summary["max_abs_error_vs_tmm"]["fdtd_1d_T"] < 5e-3
        ),
    }
    return summary


def plot_spectrum(
    case_name: str,
    wavelengths: np.ndarray,
    spectrum: dict[str, list[dict[str, float]]],
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    labels = {
        "tmm": "TMM reference",
        "rcwa_torcwa": "RCWA torcwa",
        "fdtd_1d": "1D FDTD",
    }
    styles = {
        "tmm": {"color": "black", "linewidth": 2.2},
        "rcwa_torcwa": {"color": "#7c3aed", "marker": "o"},
        "fdtd_1d": {"color": "#0f766e", "marker": "s"},
    }
    for ax, key, title in zip(axes, ["R", "T", "A"], ["Reflection", "Transmission", "Absorption"]):
        for method, rows in spectrum.items():
            ax.plot(
                wavelengths * 1000.0,
                [row[key] for row in rows],
                label=labels[method],
                **styles[method],
            )
        ax.set_title(title)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"RCWA/FDTD {case_name} alignment against TMM")
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_errors(
    wavelengths: np.ndarray,
    spectrum: dict[str, list[dict[str, float]]],
    outpath: Path,
) -> None:
    tmm = spectrum["tmm"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method, color in [("rcwa_torcwa", "#7c3aed"), ("fdtd_1d", "#0f766e")]:
        for key, linestyle in [("R", "-"), ("T", "--")]:
            errors = [
                abs(row[key] - ref[key])
                for row, ref in zip(spectrum[method], tmm)
            ]
            errors = np.maximum(np.asarray(errors, dtype=float), 1e-16)
            axes[0].semilogy(
                wavelengths * 1000.0,
                errors,
                marker="o",
                linestyle=linestyle,
                color=color,
                label=f"{method} {key}",
            )
        residual = [
            abs(row["R"] + row["T"] + row["A"] - 1.0)
            for row in spectrum[method]
        ]
        residual = np.maximum(np.asarray(residual, dtype=float), 1e-16)
        axes[1].semilogy(
            wavelengths * 1000.0,
            residual,
            marker="o",
            color=color,
            label=method,
        )
    axes[0].set_title("Absolute error vs TMM")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("abs error")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Energy residual")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("|R + T + A - 1|")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def plot_convergence(
    case_name: str,
    convergence: list[dict[str, float]],
    outpath: Path,
) -> None:
    dx = np.array([row["dx_um"] for row in convergence])
    r_error = np.maximum(np.array([row["abs_R_error"] for row in convergence]), 1e-16)
    t_error = np.maximum(np.array([row["abs_T_error"] for row in convergence]), 1e-16)
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.loglog(dx, r_error, marker="o", label="|R_FDTD - R_TMM|")
    ax.loglog(dx, t_error, marker="s", label="|T_FDTD - T_TMM|")
    ax.invert_xaxis()
    ax.set_title(f"1D FDTD {case_name} grid convergence at 550 nm")
    ax.set_xlabel("dx (um)")
    ax.set_ylabel("absolute error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def write_report(
    outdir: Path,
    case: AlignmentCase,
    fdtd_dx: float,
    fdtd_runtime: float,
    summary: dict[str, Any],
    plots: dict[str, str],
) -> None:
    loss_label = (
        "lossy dielectric stack"
        if any(abs(complex(layer.refractive_index).imag) > 0 for layer in case.layers)
        else "lossless dielectric stack"
    )
    lines = [
        "# RCWA/FDTD Alignment Report",
        "",
        f"- Structure: `{case.structure_name}`",
        f"- Incident/substrate index: `{case.incident_index}` / `{case.substrate_index}`",
        f"- Stack thickness: `{case.stack_thickness_um}` um",
        f"- FDTD dx: `{fdtd_dx}` um",
        f"- FDTD runtime c*t: `{fdtd_runtime}` um",
        f"- Reference: analytical TMM for a {loss_label}",
        "",
        "This report is a rung of the alignment ladder. It proves the",
        "normalization, source, monitor, and R/T/A conventions before moving",
        "toward trenches and the full CMOS pixel.",
        "",
        "## Layer Stack",
        "",
    ]
    for layer in case.layers:
        n_value = complex(layer.refractive_index)
        lines.append(
            f"- `{layer.name}`: n=`{n_value.real:.4g}+{n_value.imag:.4g}j`, "
            f"thickness=`{layer.thickness_um}` um"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(summary, indent=2),
            "```",
            "",
        ]
    )
    for title, filename in plots.items():
        lines.extend([f"## {title}", "", f"![{title}]({filename})", ""])
    (outdir / "alignment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    wavelengths = np.array(args.wavelengths, dtype=float)
    convergence_dx = list(args.convergence_dx)
    if args.quick:
        wavelengths = np.array([0.50, 0.55, 0.60], dtype=float)
        convergence_dx = [0.02, 0.01, 0.005]
        args.fdtd_dx = max(args.fdtd_dx, 0.005)
        args.fdtd_runtime_um = min(args.fdtd_runtime_um, 60.0)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    case = build_alignment_case(
        args.structure,
        wavelength_um=float(wavelengths[0]),
        slab_index=args.slab_index,
        slab_thickness_um=args.slab_thickness,
    )

    spectrum = run_spectrum(
        structure=args.structure,
        wavelengths=wavelengths,
        slab_index=args.slab_index,
        slab_thickness_um=args.slab_thickness,
        fdtd_dx_um=args.fdtd_dx,
        fdtd_runtime_um=args.fdtd_runtime_um,
    )
    convergence = run_fdtd_convergence(
        structure=args.structure,
        wavelength=0.55,
        slab_index=args.slab_index,
        slab_thickness_um=args.slab_thickness,
        dx_values=convergence_dx,
        runtime_um=args.fdtd_runtime_um,
    )
    summary = summarize_alignment(case, wavelengths, spectrum, convergence)

    metrics = {
        "structure": {
            "name": case.structure_name,
            "incident_index": case.incident_index,
            "substrate_index": case.substrate_index,
            "stack_thickness_um": case.stack_thickness_um,
            "layers": case.layer_summary,
        },
        "fdtd": {
            "dx_um": args.fdtd_dx,
            "runtime_um": args.fdtd_runtime_um,
            "convergence_dx_um": convergence_dx,
        },
        "summary": summary,
        "spectrum": spectrum,
        "fdtd_convergence": convergence,
    }
    (outdir / "alignment_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    plots = {
        "Spectrum Alignment": "01_spectrum_alignment.png",
        "Error Alignment": "02_alignment_errors.png",
        "FDTD Grid Convergence": "03_fdtd_grid_convergence.png",
    }
    plot_spectrum(case.structure_name, wavelengths, spectrum, outdir / plots["Spectrum Alignment"])
    plot_errors(wavelengths, spectrum, outdir / plots["Error Alignment"])
    plot_convergence(case.structure_name, convergence, outdir / plots["FDTD Grid Convergence"])
    write_report(
        outdir=outdir,
        case=case,
        fdtd_dx=args.fdtd_dx,
        fdtd_runtime=args.fdtd_runtime_um,
        summary=summary,
        plots=plots,
    )

    print(f"RCWA/FDTD alignment artifacts written to: {outdir}")
    print(f"  - {outdir / 'alignment_metrics.json'}")
    print(f"  - {outdir / 'alignment_report.md'}")
    for filename in plots.values():
        print(f"  - {outdir / filename}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
