"""Adaptive complex128 GPU accuracy runs, serialized with native fdtdz.

Single convergence runs have no warm performance claim. Resource failures never
trigger a precision downgrade. Adjacent differences are not an error bound.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"
TOLERANCE = 0.005
ORDERS = (13, 17, 21, 25, 29, 33)
TIMEOUT = 28800
TERMINAL = {"ok", "failed", "timeout", "unavailable", "process_failed"}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def delta(a, b):
    if a.get("wavelength_um") != b.get("wavelength_um"):
        raise ValueError("Cannot compare different wavelength sets")
    av, bv = a.get("rta", []), b.get("rta", [])
    if len(av) != 3 or len(bv) != 3 or any(len(v) != 3 for v in [*av, *bv]):
        raise ValueError("Expected three complete R/T/A spectra")
    differences = [
        abs(x - y) for ar, br in zip(av, bv, strict=True) for x, y in zip(ar, br, strict=True)
    ]
    if not all(math.isfinite(x) for x in differences):
        raise ValueError("Nonfinite convergence result")
    return max(differences)


def stable_tail(rows):
    return len(rows) >= 3 and all(
        delta(a, b) <= TOLERANCE for a, b in zip(rows[-3:-1], rows[-2:], strict=True)
    )


def run(name, order, grid):
    label = f"{name}_cuda_n{order}_g{grid}_convergence"
    path = OUT / f"{label}.json"
    previous = read(path)
    if previous.get("status") in TERMINAL:
        print("PRESERVE", label, previous["status"], flush=True)
        return previous
    # A convolution uses differences up to 2N, not just the mode indices N.
    if grid < 4 * order + 1:
        raise ValueError(f"epsilon grid {grid} undersamples convolution for N={order}")
    command = [
        str(ROOT / ".venv-wsl/bin/python"),
        str(ROOT / "scripts/benchmark_wsl_solvers.py"),
        "--solver",
        name,
        "--device",
        "cuda",
        "--order",
        str(order),
        "--grid",
        str(grid),
        "--repeats",
        "0",
        "--output",
        str(path),
    ]
    print("RUN", label, flush=True)
    log_path = path.with_suffix(".log")
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": ".9",
                    "JAX_ENABLE_X64": "true",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
            try:
                process.wait(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if process.returncode or not path.exists():
            raise RuntimeError(f"Process exited {process.returncode}; see {log_path.name}")
    except Exception as exc:
        failure = read(path)
        failure.update(
            solver=name,
            device="cuda",
            order=order,
            grid=grid,
            case="pattern",
            precision="complex128",
            status="timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed",
            error=str(exc),
            command=command,
        )
        write(path, failure)
    row = read(path)
    if row.get("status") == "ok":
        try:
            delta(row, row)
            if row.get("precision") != "complex128":
                raise ValueError("Unexpected precision; automatic downgrade is forbidden")
        except ValueError as exc:
            row.update(status="failed", error=str(exc))
            write(path, row)
    print("DONE", label, row.get("status"), flush=True)
    return row


def study(name):
    summary = {
        "solver": name,
        "precision": "complex128",
        "tolerance": TOLERANCE,
        "max_order": max(ORDERS),
        "per_run_timeout_s": TIMEOUT,
        "order_converged": False,
        "epsilon_grid_converged": False,
        "self_converged": False,
        "order_changes": [],
        "epsilon_grid_changes": [],
        "note": "Empirical adjacent differences are not an error bound or Meep validation.",
    }
    rows = []
    for order in ORDERS:
        if 4 * order + 1 > 128:
            break
        row = run(name, order, 128)
        if row.get("status") != "ok":
            summary["stopped_by"] = {
                "order": order,
                "grid": 128,
                "status": row.get("status"),
                "error": row.get("error"),
            }
            break
        if rows:
            summary["order_changes"].append(
                {
                    "grid": 128,
                    "from": rows[-1]["order"],
                    "to": order,
                    "max_delta": delta(rows[-1], row),
                }
            )
        rows.append(row)
        if order >= 21 and stable_tail(rows):
            summary["order_converged"] = True
            break
    if not rows:
        return summary
    selected = rows[-1]
    # Test epsilon sampling even when the order expansion hit a resource limit.
    finer = run(name, selected["order"], 256)
    if finer.get("status") != "ok":
        summary["grid_validation_failure"] = {
            "grid": 256,
            "status": finer.get("status"),
            "error": finer.get("error"),
        }
        summary["selected_file"] = f"{name}_cuda_n{selected['order']}_g128_convergence.json"
        return summary
    change = delta(selected, finer)
    summary["epsilon_grid_changes"].append(
        {"order": selected["order"], "from": 128, "to": 256, "max_delta": change}
    )
    summary["epsilon_grid_converged"] = change <= TOLERANCE
    if not summary["order_converged"] or not summary["epsilon_grid_converged"]:
        # Re-establish the last TWO order changes on one finer grid before N=33.
        fine_rows = []
        start = max(0, ORDERS.index(selected["order"]) - 2)
        for order in ORDERS[start:]:
            row = finer if order == selected["order"] else run(name, order, 256)
            if row.get("status") != "ok":
                summary["fine_order_failure"] = {
                    "order": order,
                    "grid": 256,
                    "status": row.get("status"),
                    "error": row.get("error"),
                }
                break
            if fine_rows:
                summary["order_changes"].append(
                    {
                        "grid": 256,
                        "from": fine_rows[-1]["order"],
                        "to": order,
                        "max_delta": delta(fine_rows[-1], row),
                    }
                )
            fine_rows.append(row)
            if order >= selected["order"] and stable_tail(fine_rows):
                break
        if fine_rows:
            selected = fine_rows[-1]
            summary["order_converged"] = stable_tail(fine_rows)
        # A grid128 sample at a lower N cannot validate this final finer-grid result.
        finest = run(name, selected["order"], 512)
        summary["epsilon_grid_converged"] = False
        if finest.get("status") == "ok":
            change = delta(selected, finest)
            summary["epsilon_grid_changes"].append(
                {"order": selected["order"], "from": 256, "to": 512, "max_delta": change}
            )
            summary["epsilon_grid_converged"] = change <= TOLERANCE
        else:
            summary["grid_validation_failure"] = {
                "grid": 512,
                "status": finest.get("status"),
                "error": finest.get("error"),
            }
    summary["selected_file"] = (
        f"{name}_cuda_n{selected['order']}_g{selected['grid']}_convergence.json"
    )
    summary["self_converged"] = summary["order_converged"] and summary["epsilon_grid_converged"]
    return summary


def main():
    while "DONE fdtdx_cuda_pattern_dx0.005" not in (OUT / "finish.log").read_text():
        time.sleep(20)
    while not (OUT / "fdtdz_installer_retry.log").read_text():
        time.sleep(20)
    envpath = OUT / "fdtdz_environment.json"
    if read(envpath).get("status") == "ok":
        path = OUT / "fdtdz_native.json"
        while (
            not path.exists()
            or path.stat().st_mtime < envpath.stat().st_mtime
            or read(path).get("status") not in ("ok", "failed", "unavailable")
        ):
            time.sleep(20)
    summaries = []
    with (ROOT / ".cache/wsl_gpu_validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in ["torcwa", "meent_torch", "fmmax"]:
            try:
                summaries.append(study(name))
            except Exception as exc:
                summaries.append({"solver": name, "self_converged": False, "error": str(exc)})
            write(OUT / "high_gpu_progress.json", {"status": "running", "studies": summaries})
    write(OUT / "high_gpu_finished.json", {"status": "finished", "studies": summaries})


if __name__ == "__main__":
    main()
