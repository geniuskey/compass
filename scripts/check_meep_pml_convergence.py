"""Finish Meep convergence with independent PML and matched-grid evidence.

Wait for primary studies. Adopt a wider PML only after both its width and its
spatial resolution have been checked. Extended runs are conditional, preserve
all raw results, and never turn failures into numerical measurements.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/patterned_cpu_gpu/wsl"
MEEP = Path("/home/edwin/.local/share/compass-miniconda/envs/meep/bin")
SUMMARY = OUT / "meep_pml_finished.json"
TIME_TOLERANCE = 0.0005
GRID_TOLERANCE = 0.005
PML_TOLERANCE = 0.001


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_summary(result):
    temporary = SUMMARY.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(SUMMARY)


def temporal_delta(result):
    if result.get("status") != "ok" or len(result.get("spectra", [])) != 3:
        return None
    changes = []
    for spectrum in result["spectra"]:
        series = spectrum.get("time_series", [])
        if len(series) < 3:
            return None
        changes.extend(
            float(np.max(np.abs(np.array(series[i]["rta"]) - series[i - 1]["rta"])))
            for i in [-1, -2]
        )
    return max(changes)


def compare(a, b):
    if a.get("status") != "ok" or b.get("status") != "ok":
        return None
    first = {s["wavelength_um"]: s for s in a["spectra"]}
    second = {s["wavelength_um"]: s for s in b["spectra"]}
    if first.keys() != second.keys() or len(first) != 3:
        return None
    return max(float(np.max(np.abs(np.array(first[w]["rta"]) - second[w]["rta"]))) for w in first)


def run(base, resolution, thickness, duration):
    suffix = "" if thickness == 0.7 else f"_pml{thickness}"
    if duration != base["duration"]:
        suffix += f"_t{duration:g}"
    label = f"meep_pml_validation_r{resolution}{suffix}"
    path = OUT / f"{label}.json"
    existing = read(path)
    if existing.get("status") in {"ok", "failed", "timeout"}:
        return existing, path.name
    checkpoints = [t for t in [20, 35, 55, 80, 110, 145, 185, 240, 320, 400] if t < duration]
    command = [
        str(MEEP / "mpirun"),
        "-np",
        "8",
        str(MEEP / "python"),
        str(ROOT / "scripts/benchmark_meep_reference.py"),
        "--resolution",
        str(resolution),
        "--padding",
        str(base["padding"]),
        "--pml",
        str(thickness),
        "--duration",
        str(duration),
        "--checkpoints",
        *[str(t) for t in checkpoints],
        "--adaptive-time",
        "--fwidth-factor",
        str(base["fwidth_factor"]),
        "--extinction",
        str(base["extinction"]),
        "--output",
        str(path),
    ]
    print(f"RUN {label} PML={thickness}", flush=True)
    with path.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        try:
            process.wait(timeout=86400)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            result = read(path)
            result.update(
                status="timeout", command=command, error="PML validation exceeded 24 hours"
            )
            path.write_text(json.dumps(result, indent=2))
    result = read(path)
    if result.get("status") != "timeout" and (
        process.returncode != 0 or result.get("status") != "ok"
    ):
        result.update(status="failed", command=command, returncode=process.returncode)
        path.write_text(json.dumps(result, indent=2))
    print(f"DONE {label}: {result.get('status')}", flush=True)
    return result, path.name


def ensure_time(base, resolution, thickness, primary):
    """Reuse matching primary evidence, otherwise run; extend only if needed."""
    for filename, result in primary:
        if result["resolution"] == resolution and result["pml"] == thickness:
            change = temporal_delta(result)
            if change is not None and change <= TIME_TOLERANCE:
                return result, filename
    last = ({"status": "failed"}, None)
    for duration in sorted(set([base["duration"], 320.0, 500.0])):
        if duration < base["duration"]:
            continue
        last = run(base, resolution, thickness, duration)
        change = temporal_delta(last[0])
        if last[0].get("status") != "ok" or (change is not None and change <= TIME_TOLERANCE):
            return last
    return last


def primary_reference_ready():
    """Start width validation once Meep passes, independently of RCWA jobs."""
    complete = sorted(
        [read(path) for path in OUT.glob("meep_reference_r*.json")],
        key=lambda result: result.get("resolution", 0),
    )
    complete = [r for r in complete if temporal_delta(r) is not None]
    if len(complete) < 2:
        return False
    lower, upper = complete[-2:]
    grid_delta = compare(lower, upper)
    protocol = ["padding", "pml", "extinction", "fwidth_factor", "method", "adaptive_time"]
    return (
        upper["resolution"] >= 160
        and all(lower.get(key) == upper.get(key) for key in protocol)
        and temporal_delta(lower) <= TIME_TOLERANCE
        and temporal_delta(upper) <= TIME_TOLERANCE
        and grid_delta is not None
        and grid_delta <= GRID_TOLERANCE
    )


def main():
    summary = {"status": "waiting", "accepted_base_pml": False, "accepted_reference": False}
    write_summary(summary)
    while not (OUT / "studies_finished.json").exists() and not primary_reference_ready():
        time.sleep(20)
    primary = []
    for path in OUT.glob("meep_reference_r*.json"):
        result = read(path)
        if result.get("status") == "ok" and len(result.get("spectra", [])) == 3:
            primary.append((path.name, result))
    if not primary:
        summary.update(status="failed", error="No complete primary Meep reference")
        write_summary(summary)
        return
    base_file, base = max(primary, key=lambda item: item[1]["resolution"])
    # Only reuse primary runs with the identical source and interior-domain protocol.
    primary = [
        (name, result)
        for name, result in primary
        if all(result.get(k) == base.get(k) for k in ["padding", "extinction", "fwidth_factor"])
    ]
    widths = sorted(set([base["pml"], 0.7, 1.0, 1.4, 2.0, 2.8]))
    widths = [width for width in widths if width >= base["pml"]]
    resolutions = [
        base["resolution"],
        *[r for r in [160, 200, 240, 280, 320, 400] if r > base["resolution"]],
    ]
    coarser = max(
        (r["resolution"] for _, r in primary if r["resolution"] < base["resolution"]), default=None
    )
    if coarser is None:
        summary.update(status="failed", error="No matching coarser primary resolution")
        write_summary(summary)
        return
    summary.update(
        status="running",
        base_file=base_file,
        resolution=base["resolution"],
        base_pml=base["pml"],
        tolerance=PML_TOLERANCE,
        grid_tolerance=GRID_TOLERANCE,
        time_tolerance=TIME_TOLERANCE,
        comparisons=[],
        grid_comparisons=[],
        limits={
            "maximum_resolution": max(resolutions),
            "maximum_pml": max(widths),
            "maximum_duration": 500,
        },
    )
    write_summary(summary)
    for resolution in resolutions:
        selected = None
        for first_width, second_width in pairwise(widths):
            first, first_file = ensure_time(base, resolution, first_width, primary)
            second, second_file = ensure_time(base, resolution, second_width, primary)
            change = compare(first, second)
            td = [temporal_delta(first), temporal_delta(second)]
            summary["comparisons"].append(
                {
                    "resolution": resolution,
                    "first_file": first_file,
                    "second_file": second_file,
                    "first_pml": first_width,
                    "second_pml": second_width,
                    "max_rta_delta": change,
                    "time_delta": max(td) if all(x is not None for x in td) else None,
                }
            )
            if resolution == base["resolution"] and first_width == base["pml"]:
                summary["base_to_07_delta"] = change
                summary["pml07_time_delta"] = temporal_delta(second)
                summary["diagnostic_time_delta"] = temporal_delta(second)
            write_summary(summary)
            if (
                change is not None
                and change <= PML_TOLERANCE
                and all(x is not None and x <= TIME_TOLERANCE for x in td)
            ):
                selected = first, first_file, second, second_file, first_width, change
                break
        if selected is None:
            summary.update(
                status="failed", error="No temporally converged PML pair within recorded limits"
            )
            write_summary(summary)
            return
        fine, fine_file, wide, wide_file, thickness, pml_change = selected
        lower, lower_file = ensure_time(base, coarser, thickness, primary)
        grid_change = compare(fine, lower)
        td = [temporal_delta(result) for result in [fine, lower, wide]]
        time_change = max(td) if all(x is not None for x in td) else None
        summary["grid_comparisons"].append(
            {
                "first_file": lower_file,
                "second_file": fine_file,
                "pml": thickness,
                "first_resolution": coarser,
                "second_resolution": resolution,
                "max_rta_delta": grid_change,
            }
        )
        summary.update(
            selected_reference_file=fine_file,
            matched_coarser_file=lower_file,
            width_validation_file=wide_file,
            selected_pml=thickness,
            selected_resolution=resolution,
            grid_delta=grid_change,
            time_delta=time_change,
            pml_delta=pml_change,
        )
        if (
            grid_change is not None
            and grid_change <= GRID_TOLERANCE
            and (time_change is not None and time_change <= TIME_TOLERANCE)
        ):
            summary.update(
                status="ok", accepted_reference=True, accepted_base_pml=thickness == base["pml"]
            )
            write_summary(summary)
            print(json.dumps(summary), flush=True)
            return
        write_summary(summary)
        coarser = resolution
    summary.update(
        status="failed",
        error="Matched PML grid/time convergence not attained within recorded limits",
    )
    write_summary(summary)


if __name__ == "__main__":
    main()
