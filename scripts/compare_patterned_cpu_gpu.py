"""Compare RCWA adapters and an independent Meep model on one patterned cell.

GPU runs require a verified GPU backend. CPU fallback is never counted as GPU.
Each job runs in isolation; JSON contains cold/warm timing, spectra and failures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "patterned_cpu_gpu"
PERIOD = 0.8
# Rectangles (center x/y, size x/y), in um. Edges lie on the 64-cell grid.
RECTANGLES = [(-0.2, -0.15, 0.25, 0.2), (0.175, 0.175, 0.2, 0.25)]
PILLAR_N = 2.4 + 0.02j


def make_stack():
    import numpy as np

    from compass.core.types import Layer, LayerSlice
    from compass.geometry.pixel_stack import PixelStack

    class PatternedStack(PixelStack):
        def get_permittivity_grid(self, wavelength, nx=64, ny=64, nz=64):
            z_min, z_max = self.z_range
            z = z_min + (np.arange(nz) + 0.5) * (z_max - z_min) / nz
            epsilon = np.ones((ny, nx, nz), dtype=complex)
            for layer in self.get_layer_slices(wavelength, nx, ny):
                inside = (z >= layer.z_start) & (z < layer.z_end)
                epsilon[:, :, inside] = layer.eps_grid[:, :, None]
            return epsilon

        def get_layer_slices(self, wavelength, nx=64, ny=64, n_lens_slices=8):
            x = (np.arange(nx) + 0.5) * PERIOD / nx - PERIOD / 2
            y = (np.arange(ny) + 0.5) * PERIOD / ny - PERIOD / 2
            xx, yy = np.meshgrid(x, y)
            mask = np.zeros((ny, nx), dtype=bool)
            for cx, cy, sx, sy in RECTANGLES:
                mask |= (np.abs(xx - cx) < sx / 2) & (np.abs(yy - cy) < sy / 2)
            patterned = np.where(mask, PILLAR_N**2, 1.0).astype(complex)
            return [
                LayerSlice(
                    0.0, 0.12, 0.12, np.full((ny, nx), 1.45**2, complex), "spacer", "spacer"
                ),
                LayerSlice(0.12, 0.36, 0.24, patterned, "pillars", "pillar"),
                LayerSlice(0.36, 0.44, 0.08, np.full((ny, nx), 1.6**2, complex), "cap", "cap"),
            ]

    stack = PatternedStack({"pixel": {"pitch": PERIOD, "unit_cell": [1, 1]}})
    for name, n in [("spacer", 1.45), ("pillar", PILLAR_N), ("cap", 1.6)]:
        stack.material_db.register_constant(name, n=float(n.real), k=float(n.imag))
    stack.layers = [
        Layer("spacer", 0.0, 0.12, 0.12, "spacer"),
        Layer("pillars", 0.12, 0.36, 0.24, "pillar", True),
        Layer("cap", 0.36, 0.44, 0.08, "cap"),
    ]
    # No photodiodes in this synthetic optical benchmark; QE is deliberately absent.
    stack.photodiodes = []
    return stack


def meep_run(wavelengths, resolution):
    """Independent native Meep geometry; no use of the unvalidated COMPASS adapter."""
    import meep as mp
    import numpy as np

    if not all(hasattr(mp, a) for a in ["Simulation", "Medium", "Block", "PML"]):
        raise RuntimeError("Installed meep is not MIT photonics Meep")
    mp.verbosity(0)
    spectra = []
    for wl in wavelengths:
        freq = 1.0 / wl
        padding, pml = 0.8, 0.5
        height = 0.44
        cell = mp.Vector3(PERIOD, PERIOD, height + 2 * padding + 2 * pml)
        top, bottom = height / 2, -height / 2
        source = mp.Source(
            mp.GaussianSource(freq, fwidth=0.15 * freq),
            component=mp.Ey,
            center=mp.Vector3(z=top + 0.55),
            size=mp.Vector3(PERIOD, PERIOD),
        )
        kwargs = dict(
            cell_size=cell,
            resolution=resolution,
            sources=[source],
            boundary_layers=[mp.PML(pml, direction=mp.Z)],
            k_point=mp.Vector3(),
            dimensions=3,
            eps_averaging=True,
        )
        refl_region = mp.FluxRegion(center=mp.Vector3(z=top + 0.3), size=mp.Vector3(PERIOD, PERIOD))
        trans_region = mp.FluxRegion(
            center=mp.Vector3(z=bottom - 0.3), size=mp.Vector3(PERIOD, PERIOD)
        )
        ref = mp.Simulation(**kwargs)
        rr = ref.add_flux(freq, 0, 1, refl_region)
        tr = ref.add_flux(freq, 0, 1, trans_region)
        # Fixed post-source duration is explicit and equal between reference/structure.
        # Decay/time convergence must be checked before treating this as ground truth.
        ref.run(until_after_sources=80.0)
        incident = mp.get_fluxes(tr)[0]
        reflection_data = ref.get_flux_data(rr)
        ref.reset_meep()
        if not np.isfinite(incident) or abs(incident) < 1e-20:
            raise RuntimeError("Meep reference flux is invalid")
        eps = PILLAR_N**2
        # At this single target frequency epsilon*(1+i*sigma/(2*pi*f))
        # matches the requested complex permittivity exactly.
        material = mp.Medium(
            epsilon=float(eps.real), D_conductivity=float(2 * np.pi * freq * eps.imag / eps.real)
        )
        geometry = [
            mp.Block(
                center=mp.Vector3(z=0.06 - height / 2),
                size=mp.Vector3(mp.inf, mp.inf, 0.12),
                material=mp.Medium(index=1.45),
            ),
            mp.Block(
                center=mp.Vector3(z=0.4 - height / 2),
                size=mp.Vector3(mp.inf, mp.inf, 0.08),
                material=mp.Medium(index=1.6),
            ),
        ]
        for cx, cy, sx, sy in RECTANGLES:
            geometry.append(
                mp.Block(
                    center=mp.Vector3(cx, cy, 0.24 - height / 2),
                    size=mp.Vector3(sx, sy, 0.24),
                    material=material,
                )
            )
        sim = mp.Simulation(**kwargs, geometry=geometry)
        rr = sim.add_flux(freq, 0, 1, refl_region)
        tr = sim.add_flux(freq, 0, 1, trans_region)
        sim.load_minus_flux_data(rr, reflection_data)
        sim.run(until_after_sources=80.0)
        R = -mp.get_fluxes(rr)[0] / incident
        T = mp.get_fluxes(tr)[0] / incident
        spectra.append([R, T, 1 - R - T])
        sim.reset_meep()
    return np.asarray(spectra)


def run_job(name, device, order, wavelengths, repeats, output):
    import numpy as np
    import torch

    from compass.solvers.base import SolverFactory

    torch.set_num_threads(1)
    row = {
        "solver": name,
        "requested_device": device,
        "order": order,
        "wavelength_um": wavelengths,
        "status": "failed",
        "repeats": repeats,
    }
    try:
        if name == "meep":
            if device != "cpu":
                raise RuntimeError("Meep is a CPU reference; no GPU backend")
            row["actual_device"] = "cpu"
        elif name == "fmmax":
            import jax

            target = "gpu" if device == "cuda" else "cpu"
            devices = jax.devices(target)  # raises if CUDA is absent
            jax.config.update("jax_default_device", devices[0])
            row["actual_device"] = str(jax.numpy.ones(1).device)
        elif device == "cuda":
            if name in ("grcwa", "meent_numpy"):
                raise RuntimeError(f"COMPASS {name} is CPU-only")
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA unavailable: torch={torch.__version__}")
            row["actual_device"] = str(torch.ones(1, device="cuda").device)
        else:
            row["actual_device"] = "cpu"
        stack = make_stack()
        backend = "torch" if name == "meent_torch" else "numpy"
        solver_name = "meent" if name.startswith("meent_") else name
        params = {
            "fourier_order": [order, order],
            "nG": (2 * order + 1) ** 2 + 1,
            "dtype": "complex128",
            "backend": backend,
            "grid_multiplier": 0,
            "fmm_formulation": "jones",
        }
        row["params"] = params
        row["requested_modes"] = (2 * order + 1) ** 2 if name != "meep" else None
        times = []
        previous = None
        for _ in range(repeats + 1):
            if name != "meep":
                solver = SolverFactory.create(
                    solver_name, {"name": solver_name, "params": params}, device
                )
                solver.setup_geometry(stack)
                solver.setup_source(
                    {"wavelength": {"mode": "list", "values": wavelengths}, "polarization": "TE"}
                )
            if device == "cuda":
                torch.cuda.synchronize() if name != "fmmax" else None
            start = time.perf_counter()
            if name == "meep":
                actual = meep_run(wavelengths, order)
                row["meep_resolution"] = order
            else:
                result = solver.run()
                actual = np.array([result.reflection, result.transmission, result.absorption]).T
                if result.metadata.get("failed_runs"):
                    raise RuntimeError(str(result.metadata["failed_runs"]))
            if device == "cuda" and name != "fmmax":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if not np.isfinite(actual).all():
                raise RuntimeError("NaN/Inf result")
            if previous is not None:
                row["max_repeat_delta"] = float(np.max(np.abs(actual - previous)))
            previous = actual
        row.update(
            status="ok",
            rta=actual.tolist(),
            cold_s=times[0],
            warm_s=times[1:],
            warm_median_s=statistics.median(times[1:]),
        )
        if name != "meep":
            row["metadata"] = result.metadata
        row["bounded"] = bool(np.all(actual >= -1e-6) and np.all(actual <= 1 + 1e-6))
    except Exception:
        row["error"] = traceback.format_exc()
    output.write_text(json.dumps(row, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--solvers",
        nargs="+",
        default=["torcwa", "grcwa", "meent_numpy", "meent_torch", "fmmax", "meep"],
    )
    p.add_argument("--orders", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--wavelengths", nargs="+", type=float, default=[0.45, 0.55, 0.65])
    p.add_argument("--meep-resolutions", nargs="+", type=int, default=[80, 120, 160])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--worker", nargs=4)
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    if args.report_only:
        write_report(json.loads((OUT / "metrics.json").read_text(encoding="utf-8")))
        return
    if args.worker:
        name, device, order, path = args.worker
        run_job(name, device, int(order), args.wavelengths, args.repeats, Path(path))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "geometry": {
            "period_um": PERIOD,
            "rectangles": RECTANGLES,
            "pillar_n": [PILLAR_N.real, PILLAR_N.imag],
            "layers_bottom_to_top": [["spacer", 0.12, 1.45], ["pillars", 0.24], ["cap", 0.08, 1.6]],
            "xy_grid": 64,
        },
        "conventions": "air on both sides; normal TE; R/T sums all propagating orders; no additional benchmark clipping (adapters clamp A>=0); synthetic optical structure, no QE",
        "basis_note": "torcwa/meent/fmmax use square basis; grcwa uses circular truncation with similar total count, not identical basis",
        "timing": "isolated processes; cold first call, median of subsequent calls; torch threads=1; BLAS/OMP environment threads=1; includes transfers and reference FDTD run",
        "runs": [],
    }
    for name in args.solvers:
        module = {
            "meep": "meep",
            "fmmax": "fmmax",
            "meent_numpy": "meent",
            "meent_torch": "meent",
        }.get(name, name)
        orders = args.meep_resolutions if name == "meep" else args.orders
        for device in ["cpu", "cuda"]:
            if name == "meep" and device == "cuda":
                continue
            for order in orders:
                output = OUT / f"{name}_{device}_{order}.json"
                output.unlink(missing_ok=True)
                print(f"Running {name} {device} {order}", flush=True)
                if importlib.util.find_spec(module) is None:
                    row = {
                        "solver": name,
                        "requested_device": device,
                        "order": order,
                        "status": "unavailable",
                        "error": f"Module {module} is not installed",
                    }
                else:
                    command = [
                        sys.executable,
                        __file__,
                        "--worker",
                        name,
                        device,
                        str(order),
                        str(output),
                        "--wavelengths",
                        *[str(w) for w in args.wavelengths],
                        "--repeats",
                        str(args.repeats),
                    ]
                    try:
                        result = subprocess.run(
                            command,
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            timeout=args.timeout,
                            env={
                                **os.environ,
                                "OMP_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1",
                                "JAX_ENABLE_X64": "true",
                            },
                        )
                        (OUT / f"{output.stem}.log").write_text(
                            result.stdout + result.stderr, encoding="utf-8"
                        )
                        row = (
                            json.loads(output.read_text())
                            if output.exists()
                            else {
                                "solver": name,
                                "requested_device": device,
                                "order": order,
                                "status": "process_failed",
                                "error": result.stderr,
                            }
                        )
                    except subprocess.TimeoutExpired:
                        row = {
                            "solver": name,
                            "requested_device": device,
                            "order": order,
                            "status": "timeout",
                        }
                manifest["runs"].append(row)
                (OUT / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OUT / "metrics.json", flush=True)
    write_report(manifest)


def write_report(manifest):
    """Prefer current WSL evidence; never regenerate obsolete environment claims."""
    if (OUT / "wsl" / "environment.json").exists():
        from report_wsl_solvers import write_report as current_report

        current_report()
        return
    lines = [
        "# Patterned solver measurements",
        "",
        "This report is generated from the supplied manifest only.",
        "",
        "| Solver | Requested device | Order | Status | Warm median (s) |",
        "|---|---|---:|---|---:|",
    ]
    for row in manifest["runs"]:
        lines.append(
            f"| {row['solver']} | {row['requested_device']} | {row['order']} | "
            f"{row['status']} | {row.get('warm_median_s', 'unmeasured')} |"
        )
    path = ROOT / "docs/reports/patterned-cpu-gpu-meep-comparison.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
