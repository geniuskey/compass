"""Install fdtdz in an isolated legacy-JAX/CUDA compiler environment after timings."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"
BASE = Path.home() / ".local/share/compass-miniconda"
ENV = BASE / "envs/fdtdz"


def main():
    while not (OUT / "fdtdx_cuda_dx025_timed.json").exists():
        time.sleep(20)
    result = {
        "solver": "fdtdz_install",
        "case": "diagnostic",
        "status": "failed",
        "environment": str(ENV),
    }
    with (OUT / "fdtdz_environment_setup.log").open("w") as log:
        try:
            if not (ENV / "bin/nvcc").exists():
                subprocess.run(
                    [
                        str(BASE / "bin/conda"),
                        "create",
                        "-y",
                        "-n",
                        "fdtdz",
                        "--override-channels",
                        "-c",
                        "conda-forge",
                        "python=3.12",
                        "cuda-nvcc=12.6",
                        "cuda-cudart-dev=12.6",
                        "cuda-driver-dev=12.6",
                        "cmake",
                        "pybind11",
                        "ninja",
                        "pip",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            env = {
                **os.environ,
                "CUDACXX": str(ENV / "bin/nvcc"),
                "CUDAToolkit_ROOT": str(ENV),
                "CMAKE_BUILD_PARALLEL_LEVEL": "1",
                "CXX": "/usr/bin/g++",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "UV_LINK_MODE": "copy",
            }
            subprocess.run(
                [
                    str(BASE / "bin/conda"),
                    "run",
                    "-n",
                    "fdtdz",
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(ENV / "bin/python"),
                    "--cache-dir",
                    str(ROOT / ".cache/uv-wsl"),
                    "numpy<2",
                    "jax[cuda12]==0.4.34",
                    "fdtdz==1.1.3",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            subprocess.run(
                [
                    str(BASE / "bin/conda"),
                    "run",
                    "-n",
                    "fdtdz",
                    "python",
                    "-c",
                    "import jax,fdtdz_jax; print(jax.__version__,jax.devices()); print(fdtdz_jax.fdtdz)",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
            result["status"] = "ok"
        except Exception as exc:
            result["error"] = str(exc)
    (OUT / "fdtdz_environment.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
