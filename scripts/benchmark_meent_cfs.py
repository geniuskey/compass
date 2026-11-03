"""Accuracy-only meent CFS alternative, serialized after existing GPU studies.

This is a separate algorithm variant. Default enhanced DFS measurements remain
untouched, and single-run durations are never interpreted as warm speedups.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"
PYTHON = ROOT / ".venv-wsl/bin/python"
SUMMARY = OUT / "meent_cfs_finished.json"


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def change(a, b):
    if a.get("status") != "ok" or b.get("status") != "ok":
        return None
    if a.get("wavelength_um") != b.get("wavelength_um"):
        return None
    return max(
        abs(x - y)
        for ra, rb in zip(a["rta"], b["rta"], strict=True)
        for x, y in zip(ra, rb, strict=True)
    )


def publish(summary):
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False))
    temporary.replace(SUMMARY)


def job(order, device="cuda", grid=128, *, slab=False, extinction=0.02):
    case = f"slab_k{extinction}" if slab else "pattern"
    label = f"meent_torch_cfs_{device}_{case}_n{order}_g{grid}"
    path = OUT / f"{label}.json"
    if path.exists():
        return read(path), path.name
    command = [
        str(PYTHON),
        str(ROOT / "scripts/benchmark_wsl_solvers.py"),
        "--solver",
        "meent_torch_cfs",
        "--device",
        device,
        "--order",
        str(order),
        "--grid",
        str(grid),
        "--repeats",
        "0",
        "--timing-label",
        "concurrent_accuracy_cfs",
        "--extinction",
        str(extinction),
        "--case",
        "slab" if slab else "pattern",
        "--output",
        str(path),
    ]
    print("RUN", label, flush=True)
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
            path.write_text(
                json.dumps(
                    {
                        "solver": "meent_torch_cfs",
                        "status": "timeout",
                        "command": command,
                        "error": "CFS accuracy run exceeded 4h",
                    }
                )
            )
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "solver": "meent_torch_cfs",
                    "status": "process_failed",
                    "command": command,
                    "returncode": process.returncode,
                }
            )
        )
    return read(path), path.name


def main():
    summary = {
        "status": "waiting",
        "solver_variant": "meent_torch_cfs",
        "accuracy_context": "GPU serialized; CPU accuracy jobs may run concurrently",
        "timing_repeats": 0,
        "checks": [],
        "runs": [],
    }
    publish(summary)
    while True:
        envpath, nativepath = OUT / "fdtdz_environment.json", OUT / "fdtdz_native.json"
        environment, native = read(envpath), read(nativepath)
        native_done = environment.get("status") in {"failed", "unavailable"} or (
            environment.get("status") == "ok"
            and native.get("status") in {"ok", "failed", "unavailable"}
            and nativepath.exists()
            and nativepath.stat().st_mtime >= envpath.stat().st_mtime
        )
        finish = (OUT / "finish.log").read_text() if (OUT / "finish.log").exists() else ""
        if (
            "DONE fdtdx_cuda_pattern_dx0.005" in finish
            and (OUT / "high_gpu_finished.json").exists()
            and environment
            and native_done
        ):
            break
        time.sleep(20)
    (ROOT / ".cache").mkdir(exist_ok=True)
    with (ROOT / ".cache/wsl_gpu_validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summary["status"] = "running"
        publish(summary)

        def run(*args, **kwargs):
            result, filename = job(*args, **kwargs)
            if filename not in summary["runs"]:
                summary["runs"].append(filename)
            publish(summary)
            return result, filename

        def check(kind, value, tolerance, **details):
            passed = value is not None and value <= tolerance
            summary["checks"].append(
                {
                    "kind": kind,
                    "max_rta_delta": value,
                    "tolerance": tolerance,
                    "passed": passed,
                    **details,
                }
            )
            publish(summary)
            return passed

        validated = True
        for extinction in [0.0, 0.05]:
            result, filename = run(3, slab=True, extinction=extinction)
            error = result.get("max_error_vs_tmm") if result.get("status") == "ok" else None
            validated &= check("planar_vs_tmm", error, 5e-5, file=filename)
        cpu, cpu_file = run(9, "cpu")
        gpu, gpu_file = run(9)
        validated &= check(
            "cpu_gpu_parity", change(cpu, gpu), 1e-8, first_file=cpu_file, second_file=gpu_file
        )
        for device, cfs, cfs_file in [("cpu", cpu, cpu_file), ("cuda", gpu, gpu_file)]:
            default_file = f"meent_torch_{device}_n9_g128_timed.json"
            check(
                "cfs_vs_default_dfs_n9",
                change(cfs, read(OUT / default_file)),
                0.005,
                first_file=cfs_file,
                second_file=default_file,
            )
        cfs13, cfs13_file = run(13)
        defaults13 = [
            OUT / f"meent_torch_{device}_n13_g128_convergence.json" for device in ["cuda", "cpu"]
        ]
        defaults13.append(OUT / "meent_numpy_cpu_n13_g128_convergence.json")
        default13 = next((p for p in defaults13 if read(p).get("status") == "ok"), None)
        check(
            "cfs_vs_default_dfs_n13",
            change(cfs13, read(default13)) if default13 else None,
            0.005,
            first_file=cfs13_file,
            second_file=default13.name if default13 else None,
        )
        validated &= cfs13.get("status") == "ok"
        summary["low_order_validation_passed"] = validated
        baseline_checks = [c for c in summary["checks"] if c["kind"].startswith("cfs_vs_default")]
        summary["dfs_baseline_comparisons_complete"] = all(
            c["max_rta_delta"] is not None for c in baseline_checks
        )
        summary["dfs_baseline_agreement_within_0_005"] = all(c["passed"] for c in baseline_checks)
        summary["algorithm_difference_policy"] = (
            "DFS/CFS differences >0.005 are reported, not silently treated as equivalence or "
            "as a failed CFS solve. TMM/device parity gates are mandatory; CFS must converge "
            "independently in order/grid and be compared with the validated Meep reference."
        )
        if not validated:
            summary.update(
                status="validation_failed",
                reason="CFS TMM/device parity check or N13 execution failed",
            )
            publish(summary)
            return
        finer13, finer13_file = run(13, grid=256)
        check(
            "epsilon_grid_n13",
            change(cfs13, finer13),
            0.005,
            first_file=cfs13_file,
            second_file=finer13_file,
        )
        previous, previous_file = cfs13, cfs13_file
        order_converged = False
        stable_changes = 0
        for order in [17, 21, 25, 29, 33]:
            current, current_file = run(order)
            difference = change(previous, current)
            stable = check(
                "fourier_order",
                difference,
                0.005,
                first_file=previous_file,
                second_file=current_file,
            )
            if current.get("status") != "ok":
                break
            stable_changes = stable_changes + 1 if stable else 0
            order_converged = stable_changes >= 2
            previous, previous_file = current, current_file
            if order_converged:
                break
        finer, finer_file = run(previous["order"], grid=256)
        grid_converged = check(
            "epsilon_grid_final",
            change(previous, finer),
            0.005,
            first_file=previous_file,
            second_file=finer_file,
        )
        summary.update(
            status="ok",
            order_converged=order_converged,
            grid_converged=grid_converged,
            selected_file=previous_file,
            reason="Completion is separate from scientific convergence acceptance",
        )
        publish(summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        result = read(SUMMARY)
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        publish(result)
        raise
