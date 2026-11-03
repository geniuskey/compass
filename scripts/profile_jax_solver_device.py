"""Profile the actual fmmax adapter separately from reported benchmark timings."""

from __future__ import annotations

import fcntl
import gzip
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    while not all(
        (OUT / name).exists() for name in ["high_gpu_finished.json", "fdtdx_time_finished.json"]
    ):
        time.sleep(30)
    with (ROOT / ".cache/wsl_gpu_validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        row = {
            "case": "device_profile_only",
            "profiled_solver": "fmmax",
            "status": "failed",
            "eigen_calls": [],
        }
        try:
            import jax
            from compare_patterned_cpu_gpu import make_stack

            from compass.solvers.base import SolverFactory

            original = jax.lax.linalg.eig
            seen = set()

            def observe(matrix, *args, **kwargs):
                device = getattr(matrix, "device", None)
                call = {
                    "shape": list(matrix.shape),
                    "dtype": str(matrix.dtype),
                    "device": str(device),
                }
                key = (str(matrix.shape), str(matrix.dtype), str(device))
                if key not in seen:
                    seen.add(key)
                    if getattr(device, "platform", None) == "gpu":
                        hlo = str(
                            jax.jit(lambda a: original(a, *args, **kwargs))
                            .lower(matrix)
                            .compiler_ir()
                        )
                        call["exact_input_lowering"] = [
                            line.strip() for line in hlo.splitlines() if "custom_call" in line
                        ]
                    row["eigen_calls"].append(call)
                return original(matrix, *args, **kwargs)

            jax.lax.linalg.eig = observe
            solver = SolverFactory.create(
                "fmmax",
                {
                    "name": "fmmax",
                    "params": {
                        "dtype": "complex128",
                        "fourier_order": [9, 9],
                        "epsilon_grid": 128,
                        "grid_multiplier": 0,
                        "fmm_formulation": "jones",
                    },
                },
                "cuda",
            )
            solver.setup_geometry(make_stack())
            solver.setup_source(
                {"wavelength": {"mode": "list", "values": [0.55]}, "polarization": "TE"}
            )
            solver.run()
            trace_dir = OUT / "fmmax_cuda_profile"
            with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
                result = solver.run()
            files = sorted(trace_dir.rglob("perfetto_trace.json.gz"))
            if not files:
                raise RuntimeError("JAX produced no Perfetto trace")
            with gzip.open(files[-1], "rt") as handle:
                events = json.load(handle)["traceEvents"]
            gpu_pids = {
                event.get("pid")
                for event in events
                if event.get("name") == "process_name"
                and "gpu" in str(event.get("args", {})).lower()
            }
            kernels = [
                event for event in events if event.get("pid") in gpu_pids and event.get("ph") == "X"
            ]
            row.update(
                status="ok" if kernels else "unverified",
                gpu_process_ids=sorted(gpu_pids),
                cuda_kernel_count=len(kernels),
                cuda_kernel_names=sorted({event.get("name", "") for event in kernels}),
                trace=str(files[-1].relative_to(OUT)),
                metadata=result.metadata,
                note="Profiler replay is excluded from performance timings. Exact-input lowering is recorded only for concrete GPU eigenproblem inputs; tracer observations do not independently prove device placement.",
            )
        except Exception as exc:
            row["error"] = repr(exc)
        (OUT / "jax_device_profile.json").write_text(json.dumps(row, indent=2, default=str))
        (OUT / "jax_device_profile_finished.json").write_text(json.dumps({"status": "finished"}))


if __name__ == "__main__":
    main()
