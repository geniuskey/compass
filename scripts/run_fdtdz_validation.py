"""Wait for the dedicated fdtdz environment and a free validation GPU."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"
BASE = Path.home() / ".local/share/compass-miniconda"


def main():
    path = OUT / "fdtdz_environment.json"
    while not path.exists():
        time.sleep(20)
    environment = json.loads(path.read_text())
    if environment.get("status") != "ok":
        (OUT / "fdtdz_native.json").write_text(
            json.dumps(
                {
                    "solver": "fdtdz_native",
                    "case": "lossless_slab_native",
                    "status": "unavailable",
                    "error": environment.get("error", "Environment setup failed"),
                },
                indent=2,
            )
        )
        return
    while True:
        text = (OUT / "finish.log").read_text() if (OUT / "finish.log").exists() else ""
        if "DONE fdtdx_cuda_pattern_dx0.005" in text or (OUT / "studies_finished.json").exists():
            break
        time.sleep(20)
    lock = (ROOT / ".cache/wsl_gpu_validation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    with (OUT / "fdtdz_native.log").open("w") as log:
        command = [
            str(BASE / "bin/conda"),
            "run",
            "--no-capture-output",
            "-n",
            "fdtdz",
            "python",
            str(ROOT / "scripts/benchmark_fdtdz_native.py"),
            "--output",
            str(OUT / "fdtdz_native.json"),
        ]
        try:
            started = time.time()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
            )
            try:
                process.wait(timeout=14400)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            output = OUT / "fdtdz_native.json"
            if process.returncode or not output.exists() or output.stat().st_mtime < started:
                raise RuntimeError(
                    f"Native fdtdz exited {process.returncode}; see fdtdz_native.log"
                )
        except Exception as exc:
            output = OUT / "fdtdz_native.json"
            partial = {}
            if output.exists() and output.stat().st_mtime >= started:
                with contextlib.suppress(json.JSONDecodeError, OSError):
                    partial = json.loads(output.read_text())
            partial.update(
                solver="fdtdz_native",
                case="lossless_slab_native",
                status="failed",
                error=str(exc),
            )
            output.write_text(json.dumps(partial, indent=2))


if __name__ == "__main__":
    main()
