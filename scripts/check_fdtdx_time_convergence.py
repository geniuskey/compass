"""Separate time-window convergence from FDTDX spatial discretization error."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"


def run(dx, duration):
    label = f"fdtdx_cuda_pattern_dx{dx}_t{duration}"
    path = OUT / f"{label}.json"
    if path.exists():
        return json.loads(path.read_text())
    command = [
        str(ROOT / ".venv-wsl/bin/python"),
        str(ROOT / "scripts/benchmark_wsl_solvers.py"),
        "--solver",
        "fdtdx",
        "--device",
        "cuda",
        "--spacing",
        str(dx),
        "--duration",
        str(duration),
        "--pml-layers",
        str(round(0.35 / dx)),
        "--order",
        "0",
        "--grid",
        "64",
        "--repeats",
        "0",
        "--output",
        str(path),
    ]
    with path.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "JAX_ENABLE_X64": "true",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        try:
            process.wait(timeout=28800)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            path.write_text(
                json.dumps(
                    {
                        "solver": "fdtdx",
                        "status": "timeout",
                        "command": command,
                        "error": "Temporal convergence calculation exceeded eight hours",
                    },
                    indent=2,
                )
            )
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "solver": "fdtdx",
                    "case": "diagnostic",
                    "status": "failed",
                    "error": f"time convergence process exited {process.returncode}",
                }
            )
        )
    return json.loads(path.read_text())


def main():
    while "DONE fdtdx_cuda_pattern_dx0.005" not in (OUT / "finish.log").read_text():
        time.sleep(20)
    # The native fdtdz check is short and is also a dependency of high-order
    # RCWA validation. Let it finish before a possible multi-hour fine FDTD run.
    environment_path = OUT / "fdtdz_environment.json"
    try:
        environment = json.loads(environment_path.read_text())
    except (OSError, ValueError):
        environment = {}
    if environment.get("status") == "ok":
        native_path = OUT / "fdtdz_native.json"
        while True:
            try:
                native = json.loads(native_path.read_text())
                fresh = native_path.stat().st_mtime >= environment_path.stat().st_mtime
                if fresh and native.get("status") in {"ok", "failed", "unavailable"}:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(20)
    with (ROOT / ".cache/wsl_gpu_validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(0.01, 40)
        longer = run(0.01, 160)
        base_path = OUT / "fdtdx_cuda_pattern_dx0.01.json"
        base = json.loads(base_path.read_text())
        if longer.get("status") == "ok" and base.get("status") == "ok":
            change = float(np.max(np.abs(np.array(base["rta"]) - longer["rta"])))
            print("FDTDX 80->160 at 10 nm:", change, flush=True)
            if change > 0.001:
                run(0.005, 160)
    (OUT / "fdtdx_time_finished.json").write_text(json.dumps({"status": "finished"}))


if __name__ == "__main__":
    main()
