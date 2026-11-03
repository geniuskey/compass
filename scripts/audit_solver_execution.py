"""Measure real adapter execution, separating requested and observed devices.

Run each backend/device in a fresh process because JAX and fdtd use global state.
No solver implementations are modified. Failed/NaN runs are not benchmarks.
"""

from __future__ import annotations

import argparse
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
OUT = ROOT / "outputs" / "gpu_solver_audit"


def worker(name, device, backend, output):
    import numpy as np
    import torch

    from compass.core.types import Layer
    from compass.geometry.pixel_stack import PixelStack
    from compass.solvers.base import SolverFactory
    from compass.solvers.tmm.tmm_core import transfer_matrix_1d

    torch.set_num_threads(1)
    observations = []
    # Observe the backend instance actually created, rather than its metadata.
    if name == "meent":
        import meent

        original = meent.call_mee

        def observed_call(*args, **kwargs):
            mee = original(*args, **kwargs)
            observations.append(
                {
                    "backend_id": kwargs.get("backend"),
                    "device": str(getattr(mee, "device", "numpy/cpu")),
                }
            )
            return mee

        meent.call_mee = observed_call

    original_eig = torch.linalg.eig

    def observed_eig(matrix, *args, **kwargs):
        observations.append({"torch_eig_device": str(matrix.device)})
        return original_eig(matrix, *args, **kwargs)

    torch.linalg.eig = observed_eig

    params = {
        "fourier_order": [1, 1],
        "nG": 10,
        "dtype": "complex128",
        "backend": backend,
        "grid_spacing": 0.025,
        "runtime": 20,
        "pml_layers": 12,
        "resolution": 40,
        "runtime_periods": 50,
    }
    row = {
        "solver": name,
        "requested_device": device,
        "backend": backend,
        "params": params,
        "cases": [],
        "observations": observations,
    }
    try:
        for k in [0.0, 0.05]:
            stack = PixelStack({"pixel": {"pitch": 0.5, "unit_cell": [1, 1]}})
            stack.material_db.register_constant("audit_glass", n=1.5, k=k)
            stack.layers = [Layer("slab", 0.0, 0.1, 0.1, "audit_glass")]
            if name in ("fdtd_flaport", "fdtdz", "fdtdx", "meep"):
                # Give source/flux planes a homogeneous air region on each side.
                stack.layers = [
                    Layer("air_bottom", 0.0, 0.4, 0.4, "air"),
                    Layer("slab", 0.4, 0.5, 0.1, "audit_glass"),
                    Layer("air_top", 0.5, 0.9, 0.4, "air"),
                ]
                row["air_buffer_um_per_side"] = 0.4
            times = []
            for _repeat in range(4):
                solver = SolverFactory.create(name, {"name": name, "params": params}, device=device)
                solver.setup_geometry(stack)
                solver.setup_source(
                    {
                        "wavelength": {"mode": "list", "values": [0.45, 0.55, 0.65]},
                        "polarization": "TE",
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                result = solver.run()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                actual = np.array([result.reflection, result.transmission, result.absorption]).T
                if not np.isfinite(actual).all() or result.metadata.get("failed_runs"):
                    raise RuntimeError(f"Invalid solver result: {result.metadata}")
            expected = np.array(
                [
                    transfer_matrix_1d(
                        np.array([1, 1.5 + 1j * k, 1], dtype=complex),
                        np.array([np.inf, 0.1, np.inf]),
                        wl,
                        0.0,
                        "TE",
                    )
                    for wl in result.wavelengths
                ]
            )
            grid = getattr(solver, "_last_grid", None)
            if grid is not None:
                observations.append(
                    {"fdtd_field_device": str(getattr(grid.E, "device", "numpy/cpu"))}
                )
            row["cases"].append(
                {
                    "k": k,
                    "wavelength_um": result.wavelengths.tolist(),
                    "RTA": actual.tolist(),
                    "TMM_RTA": expected.tolist(),
                    "max_abs_error_vs_tmm": float(np.max(np.abs(actual - expected))),
                    "cold_s": times[0],
                    "repeat_s": times[1:],
                    "warm_median_s": statistics.median(times[1:]),
                    "metadata": result.metadata,
                }
            )
        row["status"] = "ok"
    except Exception:
        row["status"] = "failed"
        row["error"] = traceback.format_exc()
    if name in ("fmmax", "fdtdz", "fdtdx") or backend == "jax":
        try:
            import jax

            row["jax_devices"] = [str(d) for d in jax.devices()]
            row["jax_array_device"] = str(jax.numpy.ones(1).device)
        except Exception as exc:
            row["jax_device_error"] = str(exc)
    output.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=4)
    args = parser.parse_args()
    if args.worker:
        name, device, backend, output = args.worker
        worker(name, device, backend, Path(output))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    import importlib.metadata

    import jax
    import torch

    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "jax": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "timing": "perf_counter around run(); 1 cold + 3 repeats; torch synchronized; outputs materialized to numpy; torch threads=1",
        "scope": "0.1 um n=1.5 slab in air, k=0/0.05, normal TE, 450/550/650 nm; not a patterned-pixel convergence benchmark",
        "packages": {
            p: (importlib.metadata.version(p) if importlib.util.find_spec(p) else None)
            for p in ["torcwa", "grcwa", "meent", "fmmax", "fdtd", "fdtdz", "fdtdx", "meep"]
        },
    }
    jobs = [
        (n, d, b)
        for n, b in [
            ("torcwa", "default"),
            ("grcwa", "default"),
            ("meent", "numpy"),
            ("meent", "torch"),
            ("meent", "jax"),
            ("fmmax", "default"),
            ("fdtd_flaport", "default"),
            ("fdtdz", "default"),
            ("fdtdx", "default"),
            ("meep", "default"),
        ]
        for d in ["cpu", "cuda"]
    ]
    rows = []
    for name, device, backend in jobs:
        path = OUT / f"{name}_{backend}_{device}.json"
        path.unlink(missing_ok=True)
        print(f"Running {name}/{backend}/{device}", flush=True)
        try:
            run = subprocess.run(
                [sys.executable, __file__, "--worker", name, device, backend, str(path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=150,
                env={**os.environ, "JAX_ENABLE_X64": "true", "OMP_NUM_THREADS": "1"},
            )
            (OUT / f"{path.stem}.log").write_text(run.stdout + run.stderr, encoding="utf-8")
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                rows.append(
                    {
                        "solver": name,
                        "backend": backend,
                        "requested_device": device,
                        "status": "process_failed",
                        "error": run.stderr,
                    }
                )
        except subprocess.TimeoutExpired:
            rows.append(
                {
                    "solver": name,
                    "backend": backend,
                    "requested_device": device,
                    "status": "timeout",
                }
            )
        (OUT / "metrics.json").write_text(
            json.dumps({"environment": environment, "runs": rows}, indent=2), encoding="utf-8"
        )
    print(f"Saved {OUT / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
