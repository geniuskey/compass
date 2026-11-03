"""Resume the expensive Meep grid/time study without mixing Windows data."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"
MEEP = Path("/home/edwin/.local/share/compass-miniconda/envs/meep/bin")


def main():
    # The existing 80-point job is already active. Do not duplicate it.
    path = OUT / "meep_pattern_compact_r80_t40.json"
    while not path.exists() or json.loads(path.read_text()).get("status") != "ok":
        print("Waiting for the active resolution-80 calculation", flush=True)
        time.sleep(30)
    for resolution in [120, 160]:
        path = OUT / f"meep_pattern_compact_r{resolution}_t60.json"
        if path.exists() and json.loads(path.read_text()).get("status") == "ok":
            continue
        command = [str(MEEP/"mpirun"), "-np", "8", str(MEEP/"python"),
            str(ROOT/"scripts/benchmark_meep_reference.py"), "--resolution",str(resolution),
            "--duration","60","--checkpoints","20","40","--padding","0.4",
            "--pml","0.35","--fwidth-factor","1.0","--output",str(path)]
        print("RUN " + " ".join(command),flush=True)
        with path.with_suffix('.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                           check=True, env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})


if __name__ == "__main__":
    main()
