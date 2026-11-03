"""Collect actual CUDA kernel evidence separately from benchmark timings."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
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
        import torch
        from compare_patterned_cpu_gpu import make_stack

        from compass.solvers.base import SolverFactory

        torch.set_num_threads(1)
        rows = []
        for name in ["torcwa", "meent"]:
            row = {"solver": name, "case": "device_profile_only", "status": "failed"}
            try:
                solver = SolverFactory.create(
                    name,
                    {
                        "name": name,
                        "params": {
                            "backend": "torch",
                            "dtype": "complex128",
                            "fourier_order": [9, 9],
                            "epsilon_grid": 128,
                            "grid_multiplier": 0,
                        },
                    },
                    "cuda",
                )
                solver.setup_geometry(make_stack())
                solver.setup_source(
                    {"wavelength": {"mode": "list", "values": [0.55]}, "polarization": "TE"}
                )
                solver.run()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    result = solver.run()
                    torch.cuda.synchronize()
                profile.export_chrome_trace(str(OUT / f"{name}_cuda_trace.json"))
                kernels = [
                    event
                    for event in profile.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                ]
                row.update(
                    status="ok" if kernels else "unverified",
                    cuda_kernel_count=len(kernels),
                    cuda_kernel_names=sorted({event.name for event in kernels}),
                    eigen_operations=[
                        {
                            "name": event.key,
                            "cpu_time_total_us": event.cpu_time_total,
                            "device_time_total_us": event.device_time_total,
                        }
                        for event in profile.key_averages()
                        if "eig" in event.key
                    ],
                    rta=[
                        result.reflection.tolist(),
                        result.transmission.tolist(),
                        result.absorption.tolist(),
                    ],
                    trace=f"{name}_cuda_trace.json",
                    note="Profiler replay is not a performance measurement; CUDA input alone does not establish all eigenvalue phases execute on GPU.",
                )
            except Exception as exc:
                row["error"] = repr(exc)
            rows.append(row)
            (OUT / "torch_device_profiles.json").write_text(
                json.dumps({"profiles": rows}, indent=2)
            )
        (OUT / "torch_device_profiles_finished.json").write_text(json.dumps({"status": "finished"}))


if __name__ == "__main__":
    main()
