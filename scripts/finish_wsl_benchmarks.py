"""Finish isolated timing first, then run accuracy-only studies concurrently.

Existing completed measurements and terminal failures are preserved. No older Windows data is read.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"
MEEP = Path("/home/edwin/.local/share/compass-miniconda/envs/meep/bin")
PYTHON = ROOT / ".venv-wsl/bin/python"


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def process_identity(pid):
    """Return Linux start time and state, preventing accidental PID reuse."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return fields[19], fields[0]
    except (OSError, IndexError):
        return None


def existing_worker(command, path):
    """Match an exact output argument and executable; MPI ranks are excluded."""
    matches = []
    expected_executable = Path(command[0]).resolve()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = [x.decode() for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
            if not args or Path(args[0]).resolve() != expected_executable:
                continue
            if "--output" not in args:
                continue
            if Path(args[args.index("--output") + 1]).resolve() != path.resolve():
                continue
            # Also require the same benchmark script, not merely a shared output path.
            scripts = [x for x in command if x.endswith(".py")]
            if not scripts or not all(x in args for x in scripts):
                continue
            identity = process_identity(int(proc.name))
            if identity and identity[1] != "Z":
                matches.append((int(proc.name), identity[0]))
        except (OSError, UnicodeDecodeError, IndexError):
            continue
    if len(matches) > 1:
        raise RuntimeError(f"Multiple active workers match {path}: {matches}")
    return matches[0] if matches else None


def wait_existing(worker, timeout):
    """Wait for a surviving child of an older controller without relaunching it."""
    pid, start_time = worker
    deadline = time.monotonic() + timeout
    while True:
        identity = process_identity(pid)
        if identity is None or identity[0] != start_time or identity[1] == "Z":
            return
        if time.monotonic() >= deadline:
            # Only the positively identified benchmark process group is terminated.
            os.killpg(pid, signal.SIGTERM)
            raise subprocess.TimeoutExpired([f"resumed PID {pid}"], timeout)
        time.sleep(10)


def run(label, command, timeout=7200):
    path = OUT / f"{label}.json"
    if read(path).get("status") in {"ok", "failed", "timeout", "unavailable"}:
        return read(path)
    print(f"RUN {label}", flush=True)
    try:
        worker = existing_worker(command, path)
        if worker:
            print(f"RESUME {label}: existing PID {worker[0]}", flush=True)
            wait_existing(worker, timeout)
        else:
            with path.with_suffix(".log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "JAX_ENABLE_X64": "true",
                    },
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
    except subprocess.TimeoutExpired:
        result = read(path)
        result.update(status="timeout", command=command)
        path.write_text(json.dumps(result, indent=2))
    except Exception as exc:
        result = read(path)
        result.update(status="failed", error=str(exc), command=command)
        path.write_text(json.dumps(result, indent=2))
    result = read(path)
    if result.get("status") not in {"ok", "failed", "timeout", "unavailable"}:
        result.update(
            status="failed", error="Process exited without a completed result JSON", command=command
        )
        path.write_text(json.dumps(result, indent=2))
    print(f"DONE {label}: {result.get('status', 'process_failed')}", flush=True)
    return result


def adapter(name, device, order=13, grid=128, label=None, **kwargs):
    label = label or f"{name}_{device}_n{order}_g{grid}_convergence"
    command = [
        str(PYTHON),
        str(ROOT / "scripts/benchmark_wsl_solvers.py"),
        "--solver",
        name,
        "--device",
        device,
        "--order",
        str(order),
        "--grid",
        str(grid),
        "--repeats",
        "0",
        "--output",
        str(OUT / f"{label}.json"),
    ]
    for k, v in kwargs.items():
        command += [
            "--" + k.replace("_", "-"),
            *([str(x) for x in v] if isinstance(v, list) else [str(v)]),
        ]
    # High-order RCWA and fine FDTD can require hours on the available hardware.
    # Keep a generous bound instead of treating a one-hour cutoff as invalid physics.
    timeout = 28800
    return run(label, command, timeout=timeout)


def meep_reference():
    import numpy as np

    previous = None
    for resolution in [120, 160, 200, 240]:
        label = f"meep_reference_r{resolution}"
        command = [
            str(MEEP / "mpirun"),
            "-np",
            "8",
            str(MEEP / "python"),
            str(ROOT / "scripts/benchmark_meep_reference.py"),
            "--resolution",
            str(resolution),
            "--padding",
            ".4",
            "--pml",
            ".35",
            "--duration",
            "185",
            "--checkpoints",
            "20",
            "35",
            "55",
            "80",
            "110",
            "145",
            "--adaptive-time",
            "--fwidth-factor",
            ".15",
            "--output",
            str(OUT / f"{label}.json"),
        ]
        result = run(label, command, timeout=43200)
        if result.get("status") != "ok":
            return
        values = np.array([r["rta"] for r in result["spectra"]])
        time_change = max(
            np.max(np.abs(np.array(s["time_series"][i]["rta"]) - s["time_series"][i - 1]["rta"]))
            for s in result["spectra"]
            for i in [-1, -2]
        )
        grid_change = float(np.max(np.abs(values - previous))) if previous is not None else None
        print(f"MEEP convergence r={resolution}: time={time_change} grid={grid_change}", flush=True)
        if (
            resolution >= 160
            and grid_change is not None
            and grid_change <= 0.005
            and time_change <= 0.0005
        ):
            return
        previous = values


def rcwa_convergence():
    import numpy as np

    for name in ["torcwa", "grcwa", "meent_numpy", "fmmax"]:
        r13 = adapter(name, "cpu", 13, 128)
        if r13.get("status") != "ok":
            continue
        r17 = adapter(name, "cpu", 17, 128)
        for grid in [64, 256]:
            adapter(name, "cpu", 13, grid)
        if r17.get("status") == "ok":
            previous, previous_grid = r17, 128
            changes = [float(np.max(np.abs(np.array(r13["rta"]) - r17["rta"])))]
            for order in [21, 25, 29, 33]:
                if len(changes) >= 2 and max(changes[-2:]) <= 0.005:
                    break
                # Circular grcwa bases extend farther along each axis than a
                # square basis with the same nominal N. Avoid FFT aliasing.
                grid = 256 if order >= 29 else 128
                if grid != previous_grid:
                    previous = adapter(name, "cpu", previous["order"], grid)
                    if previous.get("status") != "ok":
                        break
                    changes = []
                    previous_grid = grid
                current = adapter(name, "cpu", order, grid)
                if current.get("status") != "ok":
                    break
                changes.append(float(np.max(np.abs(np.array(previous["rta"]) - current["rta"]))))
                previous = current
            # Sampling convergence at the highest completed order is distinct
            # from the N=13 sampling study above.
            if previous.get("status") == "ok":
                adapter(name, "cpu", previous["order"], previous_grid * 2)


def gpu_validation():
    for name in ["torcwa", "meent_torch", "meent_jax", "fmmax"]:
        adapter(
            name, "cuda", 1, 64, label=f"{name}_cuda_slab_absorbing", case="slab", extinction=0.05
        )
    for extinction in [0.0, 0.05]:
        adapter(
            "fdtdx",
            "cuda",
            0,
            64,
            label=f"fdtdx_cuda_slab_dx0125_k{extinction}",
            case="slab",
            extinction=extinction,
            spacing=0.0125,
            pml_layers=28,
            duration=80,
        )
    for device in ["cpu", "cuda"]:
        adapter(
            "fdtd_flaport",
            device,
            0,
            64,
            label=f"flaport_{device}_slab_final",
            case="slab",
            extinction=0.0,
            wavelengths=[0.55],
        )
    for dx, pml in [(0.0125, 28), (0.01, 35), (0.005, 70)]:
        adapter(
            "fdtdx",
            "cuda",
            0,
            64,
            label=f"fdtdx_cuda_pattern_dx{dx}",
            spacing=dx,
            pml_layers=pml,
            duration=80,
        )


def main():
    print("Waiting for isolated timing to finish", flush=True)
    while not (OUT / "fdtdx_cuda_dx025_timed.json").exists():
        time.sleep(20)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(f) for f in [meep_reference, rcwa_convergence, gpu_validation]]
        for future in futures:
            future.result()
    subprocess.run([str(PYTHON), str(ROOT / "scripts/report_wsl_solvers.py")], cwd=ROOT, check=True)
    (OUT / "studies_finished.json").write_text(json.dumps({"status": "finished"}))


if __name__ == "__main__":
    main()
