"""Isolated, resumable WSL measurements; never merge older Windows measurements.

Timing includes adapter run(), all NumPy result transfers and synchronization.
Device observations hook actual eigenproblems, not a demonstration GPU tensor.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"


def fdtdx_progress_wrapper(solve, path, context):
    """Persist each completed wavelength without treating partial data as a run.

    This wrapper is only installed for convergence studies, never isolated
    performance measurements. Native solve materializes fluxes before return,
    so the completion event also follows actual GPU synchronization/transfer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(event):
        with path.open("a") as stream:
            stream.write(json.dumps(event, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print("FDTDX_PROGRESS " + json.dumps(event, default=str), flush=True)

    def wrapped(epsilon, wavelength, spacing, duration, pml_cells, device, polarization, **kw):
        shape = list(epsilon.shape)
        padding = max(2, round(0.8 / spacing))
        courant = kw.get("courant_factor", 0.5)
        event = {
            **context,
            "event": "wavelength_started",
            "wavelength_um": wavelength,
            "requested_device": device,
            "polarization": polarization,
            "grid_spacing_um": spacing,
            "duration_um_over_c": duration,
            "pml_layers": pml_cells,
            "precision": kw.get("precision", "float64"),
            "interior_grid_size_yxz": shape,
            "total_grid_size_xyz": [shape[1], shape[0], shape[2] + 2 * (padding + pml_cells)],
            "air_padding_um": padding * spacing,
            "courant_factor": courant,
            "time_steps": round(duration / (spacing * courant / math.sqrt(3))),
            "api": "place_objects/run_fdtd/PhasorPoyntingFluxDetector",
            "wall_time_unix": time.time(),
        }
        emit(event)
        started = time.perf_counter()
        try:
            result = solve(
                epsilon, wavelength, spacing, duration, pml_cells, device, polarization, **kw
            )
            if len(result) != 4 or not all(math.isfinite(float(x)) for x in result[:3]):
                raise RuntimeError(f"Invalid native FDTDX result: {result}")
            event.update(
                event="wavelength_completed",
                rta=[float(x) for x in result[:3]],
                actual_device=str(result[3]),
            )
        except Exception:
            event.update(event="wavelength_failed", error=traceback.format_exc())
            raise
        finally:
            event.update(elapsed_s=time.perf_counter() - started, wall_time_unix=time.time())
            emit(event)
        return result

    return wrapped


def environment():
    import jax
    import numpy as np
    import torch

    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "jax": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "packages": {},
    }
    for p in (
        "torch",
        "jax",
        "jaxlib",
        "jax-cuda12-plugin",
        "numpy",
        "torcwa",
        "grcwa",
        "meent",
        "fmmax",
        "fdtd",
        "fdtdz",
        "fdtdx",
        "optax",
    ):
        try:
            data["packages"][p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            data["packages"][p] = None
    data["nvidia_smi"] = subprocess.run(
        ["/usr/lib/wsl/lib/nvidia-smi"], capture_output=True, text=True
    ).stdout
    if torch.cuda.is_available():
        a = torch.arange(1024.0, device="cuda").reshape(32, 32)
        b = a @ a.T
        torch.cuda.synchronize()
        data["torch_test"] = {"device": str(b.device), "sum": b.sum().item()}
        data["gpu_name"] = torch.cuda.get_device_name()
    try:
        import jax.numpy as jnp

        with jax.default_device(jax.devices("gpu")[0]):
            a = jnp.arange(1024.0).reshape(32, 32)
            b = (a @ a.T).block_until_ready()
            data["jax_test"] = {"device": str(b.device), "sum": float(np.asarray(b).sum())}
    except Exception:
        data["jax_gpu_error"] = traceback.format_exc()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "environment.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


def worker(args):
    import numpy as np
    import torch
    from compare_patterned_cpu_gpu import make_stack

    from compass.solvers.base import SolverFactory

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    observations = []
    row = {
        "solver": args.solver,
        "device": args.device,
        "order": args.order,
        "grid": args.grid,
        "case": args.case,
        "extinction": args.extinction,
        "wavelength_um": args.wavelengths,
        "status": "failed",
        "observations": observations,
        "precision": "float64+complex128_dft" if args.solver == "fdtdx" else "complex128",
        "timing_context": args.timing_label,
        "threads": 1,
        "repeats": args.repeats,
    }

    def record(**kw):
        if kw not in observations:
            observations.append(kw)

    original = torch.linalg.eig

    def torch_eig(a, *rest, **kw):
        record(operation="torch.linalg.eig", device=str(a.device), shape=list(a.shape))
        return original(a, *rest, **kw)

    torch.linalg.eig = torch_eig
    original_np = np.linalg.eig

    def np_eig(a, *rest, **kw):
        record(operation="numpy.linalg.eig", device="cpu", shape=list(a.shape))
        return original_np(a, *rest, **kw)

    np.linalg.eig = np_eig
    try:
        name = args.solver.removesuffix("_cfs")
        cfs_variant = args.solver.endswith("_cfs")
        if cfs_variant and not name.startswith("meent_"):
            raise ValueError("The _cfs variant is supported only for meent backends")
        jax_backend = name in ("fmmax", "meent_jax", "fdtdx", "fdtdz")
        if jax_backend:
            import jax
            import jax.numpy as jnp

            target = "gpu" if args.device == "cuda" else "cpu"
            selected = jax.devices(target)[0]
            jax.config.update("jax_default_device", selected)
            row["jax_device"] = str(selected)
            original_jax = jnp.linalg.eig

            def jax_eig(a, *rest, **kw):
                record(
                    operation="jax.numpy.linalg.eig",
                    device=str(getattr(a, "device", "tracer")),
                    shape=list(a.shape),
                )
                return original_jax(a, *rest, **kw)

            jnp.linalg.eig = jax_eig
            original_lax = jax.lax.linalg.eig

            def lax_eig(a, *rest, **kw):
                record(
                    operation="jax.lax.linalg.eig",
                    device=str(getattr(a, "device", "tracer")),
                    shape=list(a.shape),
                    implementation=str(kw.get("implementation", "automatic")),
                    left=kw.get("compute_left_eigenvectors", True),
                )
                return original_lax(a, *rest, **kw)

            jax.lax.linalg.eig = lax_eig
            hlo = str(
                jax.jit(lambda a: original_lax(a, compute_left_eigenvectors=False))
                .lower(jnp.eye(2, dtype=jnp.complex128))
                .compiler_ir()
            )
            row["jax_eig_lowering"] = [
                line.strip() for line in hlo.splitlines() if "custom_call" in line or "geev" in line
            ]

        if args.device == "cuda" and not jax_backend:
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA unavailable")
            torch.cuda.reset_peak_memory_stats()
        stack = make_stack()
        if args.case == "slab":
            from compass.core.types import Layer
            from compass.geometry.pixel_stack import PixelStack

            stack = PixelStack({"pixel": {"pitch": 0.8, "unit_cell": [1, 1]}})
            stack.material_db.register_constant("glass", n=1.5, k=args.extinction)
            stack.layers = [Layer("slab", 0, 0.1, 0.1, "glass")]
            stack.photodiodes = []
        else:
            import compare_patterned_cpu_gpu as pattern

            pattern.PILLAR_N = 2.4 + 1j * args.extinction
            stack = make_stack()
        if name in ("fdtd_flaport", "fdtdz"):
            from compass.core.types import Layer

            lo, hi = stack.z_range
            stack.layers = [
                Layer("air_bottom", lo - 0.8, lo, 0.8, "air"),
                *stack.layers,
                Layer("air_top", hi, hi + 0.8, 0.8, "air"),
            ]
        backend = name.split("_", 1)[1] if name.startswith("meent_") else "numpy"
        solver_name = "meent" if name.startswith("meent_") else name
        params = {
            "fourier_order": [args.order] * 2,
            "nG": (2 * args.order + 1) ** 2 + 1,
            "dtype": "complex128",
            "backend": backend,
            "grid_multiplier": 0,
            "epsilon_grid": args.grid,
            "fmm_formulation": "jones",
            "grid_spacing": args.spacing,
            "runtime": args.runtime,
            "duration": args.duration,
            "pml_layers": args.pml_layers,
            "resolution": 80,
            "runtime_periods": 100,
        }
        if cfs_variant:
            params.update(fourier_type=1, enhanced_dfs=True)
            row["algorithm_variant"] = "meent_raster_cfs"
        row["params"] = params
        if solver_name == "grcwa":
            import grcwa

            obj = grcwa.obj(params["nG"], [0.8, 0], [0, 0.8], 1 / 0.55, 0, 0, verbose=0)
            obj.Add_LayerUniform(0, 1)
            obj.Add_LayerUniform(0, 1)
            obj.Init_Setup()
            row["actual_modes"] = obj.nG
            row["basis"] = "circular"
        else:
            row["actual_modes"] = (
                (2 * args.order + 1) ** 2
                if solver_name not in ("fdtd_flaport", "fdtdz", "fdtdx")
                else None
            )
            row["basis"] = "square"
        progress_context = {"solver": name, "case": args.case, "pid": os.getpid()}
        if name == "fdtdx" and args.timing_label == "concurrent_convergence":
            from compass.solvers.fdtd import fdtdx_native

            progress_path = args.output.parent / "progress" / f"{args.output.stem}.jsonl"
            fdtdx_native.solve = fdtdx_progress_wrapper(
                fdtdx_native.solve, progress_path, progress_context
            )
            row["progress_jsonl"] = str(progress_path)
        times, values = [], []
        for repeat in range(args.repeats + 1):
            progress_context["repeat"] = repeat
            solver = SolverFactory.create(
                solver_name, {"name": solver_name, "params": params}, args.device
            )
            solver.setup_geometry(stack)
            solver.setup_source(
                {"wavelength": {"mode": "list", "values": args.wavelengths}, "polarization": "TE"}
            )
            if args.device == "cuda" and not jax_backend:
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = solver.run()
            actual = np.array([result.reflection, result.transmission, result.absorption]).T
            if name == "fdtd_flaport" and getattr(solver, "_last_grid", None) is not None:
                record(operation="fdtd_field", device=str(solver._last_grid.E.device))
            if args.device == "cuda" and not jax_backend:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if not np.isfinite(actual).all() or result.metadata.get("failed_runs"):
                raise RuntimeError(f"Invalid result: {result.metadata}")
            values.append(actual)
            row.update(
                cold_s=times[0], warm_s=times[1:], rta=actual.tolist(), metadata=result.metadata
            )
            print(
                f"{name} {args.device} N={args.order} grid={args.grid}: repeat {repeat} {times[-1]:.3f}s {actual.tolist()}",
                flush=True,
            )
        row.update(
            status="ok",
            warm_median_s=statistics.median(times[1:]) if len(times) > 1 else None,
            max_repeat_delta=float(np.max(np.abs(np.array(values) - values[0]))),
        )
        if args.device == "cuda" and not jax_backend:
            row["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        elif jax_backend:
            row["jax_memory_stats"] = selected.memory_stats()
        if args.case == "slab":
            from compass.solvers.tmm.tmm_core import transfer_matrix_1d

            expected = np.asarray(
                [
                    transfer_matrix_1d(
                        np.array([1, 1.5 + 1j * args.extinction, 1]),
                        np.array([np.inf, 0.1, np.inf]),
                        wl,
                        0.0,
                        "TE",
                    )
                    for wl in args.wavelengths
                ]
            )
            row["tmm_rta"] = expected.tolist()
            row["max_error_vs_tmm"] = float(np.max(np.abs(actual - expected)))
        row["residual_A_delta"] = float(
            np.max(np.abs(actual[:, 2] - (1 - actual[:, 0] - actual[:, 1])))
        )
    except Exception:
        row["error"] = traceback.format_exc()
        print(row["error"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, default=str))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--environment", action="store_true")
    p.add_argument("--jobs", type=Path)
    p.add_argument("--solver", default="torcwa")
    p.add_argument("--device", default="cpu")
    p.add_argument("--order", type=int, default=9)
    p.add_argument("--pml-layers", type=int, default=12)
    p.add_argument("--timing-label", default="concurrent_convergence")
    p.add_argument("--spacing", type=float, default=0.025)
    p.add_argument("--duration", type=float, default=80.0)
    p.add_argument("--runtime", type=float, default=20.0)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--case", default="pattern", choices=["pattern", "slab"])
    p.add_argument("--extinction", type=float, default=0.02)
    p.add_argument("--wavelengths", nargs="+", type=float, default=[0.45, 0.55, 0.65])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.jobs:
        jobs = json.loads(args.jobs.read_text())
        OUT.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            label = job.pop("label")
            output = OUT / f"{label}.json"
            if output.exists():
                print(f"Preserving existing {output}", flush=True)
                continue
            cmd = [sys.executable, __file__, "--output", str(output)]
            for key, value in job.items():
                cmd += [
                    "--" + key.replace("_", "-"),
                    *([str(x) for x in value] if isinstance(value, list) else [str(value)]),
                ]
            print("RUN " + " ".join(cmd), flush=True)
            with (OUT / f"{label}.log").open("w") as log:
                try:
                    run = subprocess.run(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=3600,
                        env={
                            **os.environ,
                            "JAX_PLATFORMS": "cpu" if job.get("device") == "cpu" else "cuda,cpu",
                        },
                    )
                    if not output.exists():
                        output.write_text(
                            json.dumps(
                                {**job, "status": "process_failed", "returncode": run.returncode},
                                indent=2,
                            )
                        )
                except subprocess.TimeoutExpired:
                    output.write_text(json.dumps({**job, "status": "timeout"}, indent=2))
    elif args.environment:
        environment()
    else:
        worker(args)


if __name__ == "__main__":
    main()
