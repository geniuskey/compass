"""Native Meep reference with explicit flux subtraction and convergence controls.

Run in the conda pymeep environment; this worker does not depend on COMPASS,
PyTorch, or JAX. Each wavelength is saved immediately, including raw signed flux.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


def atomic_result_write(path, result):
    """Publish a complete JSON snapshot without exposing a truncated checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def result_protocol(args):
    """All inputs needed to decide whether completed wavelengths can be reused."""
    return dict(
        checkpoint_schema=1,
        case=args.case,
        resolution=args.resolution,
        duration=args.duration,
        extinction=args.extinction,
        padding=args.padding,
        pml=args.pml,
        method="solve_cw" if args.cw else "time_domain",
        fwidth_factor=args.fwidth_factor,
        adaptive_time=args.adaptive_time,
        checkpoints=sorted(set(args.checkpoints)),
        wavelengths=list(args.wavelengths),
    )


def resumable_result(path, protocol):
    """Reuse only a validated completed prefix from exactly the same protocol.

    Earlier files omitted checkpoint/source-list metadata. They remain readable by
    reports, but cannot safely be resumed and are never overwritten here.
    """
    path = Path(path)
    if not path.exists():
        return {**protocol, "status": "running", "spectra": []}
    saved = json.loads(path.read_text())
    if not isinstance(saved, dict):
        raise ValueError(f"Checkpoint {path} is not a result object; preserving it")
    differences = [key for key, value in protocol.items() if saved.get(key) != value]
    if differences:
        raise ValueError(
            f"Checkpoint protocol differs or is incomplete ({', '.join(differences)}); "
            f"preserving {path}. Use a new output path for this calculation."
        )
    if saved.get("status") not in {"running", "ok"}:
        raise ValueError(f"Checkpoint has non-resumable status; preserving {path}")
    rows = saved.get("spectra")
    wavelengths = protocol["wavelengths"]
    if not isinstance(rows, list) or len(rows) > len(wavelengths):
        raise ValueError(f"Invalid completed wavelength list; preserving {path}")
    for index, row in enumerate(rows):
        try:
            valid = (
                row["wavelength_um"] == wavelengths[index]
                and row["method"] == protocol["method"]
                and row["fwidth_factor"] == protocol["fwidth_factor"]
                and len(row["rta"]) == 3
                and all(math.isfinite(value) for value in row["rta"])
                and row["time_series"]
                and all(
                    len(point["rta"]) == 3
                    and all(math.isfinite(value) for value in point["rta"])
                    for point in row["time_series"]
                )
                and all(
                    math.isfinite(row["raw_flux"][key])
                    for key in ("incident", "incident_above", "reflected", "transmitted")
                )
                and abs(row["raw_flux"]["incident"]) >= 1e-20
                and math.isfinite(row["final_time"])
                and math.isfinite(row["reference_final_time"])
                and math.isfinite(row["elapsed_s"])
                and row["meep_version"]
                and row["mpi_processes"] >= 1
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"Incomplete or invalid wavelength row {index}; preserving {path}")
    if saved["status"] == "ok" and len(rows) != len(wavelengths):
        raise ValueError(f"Completed checkpoint is missing wavelengths; preserving {path}")
    return saved


def collective_resume(mp, path, protocol):
    """Make every MPI rank use the master's single immutable checkpoint read."""
    if mp.count_processors() == 1:
        return resumable_result(path, protocol)
    comm = getattr(mp, "comm", None)
    if comm is None:
        raise RuntimeError("MPI checkpoint agreement requires Meep's mpi4py communicator")
    payload = None
    if mp.am_master():
        try:
            payload = {"result": resumable_result(path, protocol)}
        except Exception as error:
            payload = {"error": str(error)}
    payload = comm.bcast(payload, root=0)
    if "error" in payload:
        raise ValueError(payload["error"])
    return payload["result"]


def compute(
    wavelength,
    resolution,
    duration,
    case="pattern",
    extinction=0.02,
    padding=0.8,
    pml=0.5,
    fwidth_factor=0.15,
    checkpoints=(),
    cw=False,
    adaptive_time=False,
):
    import meep as mp
    import numpy as np

    assert all(hasattr(mp, a) for a in ("Simulation", "Medium", "Vector3", "Block", "PML"))
    mp.verbosity(0)
    freq = 1 / wavelength
    height = 0.44 if case == "pattern" else 0.1
    # A 1D slab has exactly the same normal-incidence solution as a 3D slab.
    slab = case == "slab"
    period = 0.8
    cell = mp.Vector3(0 if slab else period, 0 if slab else period, height + 2 * (padding + pml))
    plane = mp.Vector3(0 if slab else period, 0 if slab else period)
    top, bottom = height / 2, -height / 2
    source = mp.Source(
        mp.ContinuousSource(freq) if cw else mp.GaussianSource(freq, fwidth=fwidth_factor * freq),
        component=mp.Ex if slab else mp.Ey,
        center=mp.Vector3(z=top + padding * 0.7),
        size=plane,
    )
    kwargs = dict(
        cell_size=cell,
        resolution=resolution,
        sources=[source],
        boundary_layers=[mp.PML(pml, direction=mp.Z)],
        k_point=mp.Vector3(),
        dimensions=1 if slab else 3,
        eps_averaging=True,
        force_complex_fields=cw,
    )
    refl = mp.FluxRegion(center=mp.Vector3(z=top + padding * 0.35), size=plane)
    trans = mp.FluxRegion(center=mp.Vector3(z=bottom - padding * 0.35), size=plane)

    def medium(n):
        eps = complex(n) ** 2
        return mp.Medium(epsilon=eps.real, D_conductivity=2 * np.pi * freq * eps.imag / eps.real)

    if slab:
        geometry = [
            mp.Block(
                center=mp.Vector3(),
                size=mp.Vector3(mp.inf, mp.inf, 0.1),
                material=medium(1.5 + 1j * extinction),
            )
        ]
    else:
        geometry = [
            mp.Block(
                center=mp.Vector3(z=0.06 - height / 2),
                size=mp.Vector3(mp.inf, mp.inf, 0.12),
                material=medium(1.45),
            ),
            mp.Block(
                center=mp.Vector3(z=0.4 - height / 2),
                size=mp.Vector3(mp.inf, mp.inf, 0.08),
                material=medium(1.6),
            ),
        ]
        for cx, cy, sx, sy in [(-0.2, -0.15, 0.25, 0.2), (0.175, 0.175, 0.2, 0.25)]:
            geometry.append(
                mp.Block(
                    center=mp.Vector3(cx, cy, 0.24 - height / 2),
                    size=mp.Vector3(sx, sy, 0.24),
                    material=medium(2.4 + 1j * extinction),
                )
            )
    start = time.perf_counter()
    ref = mp.Simulation(**kwargs)
    rr = ref.add_flux(freq, 0, 1, refl)
    tr = ref.add_flux(freq, 0, 1, trans)
    if cw:
        ref.init_sim()
        if ref.solve_cw(tol=1e-9, maxiters=500, L=2) is False:
            raise RuntimeError("Meep empty-cell CW solve did not converge")
    else:
        ref.run(until_after_sources=min(duration, 10.0))
    incident = mp.get_fluxes(tr)[0]
    incident_above = mp.get_fluxes(rr)[0]
    ref_time = ref.meep_time()
    data = ref.get_flux_data(rr)
    ref.reset_meep()
    if not np.isfinite(incident) or abs(incident) < 1e-20:
        raise RuntimeError("Invalid reference flux")
    sim = mp.Simulation(**kwargs, geometry=geometry)
    rr = sim.add_flux(freq, 0, 1, refl)
    tr = sim.add_flux(freq, 0, 1, trans)
    sim.load_minus_flux_data(rr, data)
    time_series = []
    elapsed_duration = 0.0
    if cw:
        sim.init_sim()
        for tolerance in [1e-6, 1e-8, 1e-9]:
            mp.scale_flux_fields(0, rr)
            mp.scale_flux_fields(0, tr)
            sim.load_minus_flux_data(rr, data)
            if sim.solve_cw(tol=tolerance, maxiters=500, L=2) is False:
                raise RuntimeError(f"Meep CW solve failed at tolerance {tolerance}")
            reflected, transmitted = mp.get_fluxes(rr)[0], mp.get_fluxes(tr)[0]
            R, T = -reflected / incident, transmitted / incident
            time_series.append(dict(tolerance=tolerance, rta=[R, T, 1 - R - T]))
            if mp.am_master():
                print(f"Meep CW wl={wavelength} res={resolution}: {time_series[-1]}", flush=True)
    else:
        for checkpoint in sorted(set([*checkpoints, duration])):
            if checkpoint > duration:
                continue
            if not time_series:
                sim.run(until_after_sources=checkpoint)
            else:
                sim.run(until=checkpoint - elapsed_duration)
            elapsed_duration = checkpoint
            reflected, transmitted = mp.get_fluxes(rr)[0], mp.get_fluxes(tr)[0]
            R, T = -reflected / incident, transmitted / incident
            time_series.append(
                dict(duration=checkpoint, rta=[R, T, 1 - R - T], time=sim.meep_time())
            )
            if mp.am_master():
                print(
                    f"Meep wl={wavelength} res={resolution} post-source={checkpoint}: {time_series[-1]}",
                    flush=True,
                )
            if adaptive_time and len(time_series) >= 4:
                changes = [
                    np.max(np.abs(np.array(time_series[i]["rta"]) - time_series[i - 1]["rta"]))
                    for i in [-1, -2]
                ]
                if max(changes) <= 5e-4:
                    break
    reflected, transmitted = mp.get_fluxes(rr)[0], mp.get_fluxes(tr)[0]
    R, T = -reflected / incident, transmitted / incident
    row = dict(
        wavelength_um=wavelength,
        rta=[R, T, 1 - R - T],
        raw_flux=dict(
            incident=incident,
            incident_above=incident_above,
            reflected=reflected,
            transmitted=transmitted,
        ),
        method="solve_cw" if cw else "time_domain",
        time_series=time_series,
        fwidth_factor=fwidth_factor,
        final_time=sim.meep_time(),
        reference_final_time=ref_time,
        elapsed_s=time.perf_counter() - start,
        meep_version=mp.__version__,
        mpi_processes=mp.count_processors(),
    )
    sim.reset_meep()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=int, default=80)
    p.add_argument("--adaptive-time", action="store_true")
    p.add_argument("--cw", action="store_true")
    p.add_argument("--slab-suite", action="store_true")
    p.add_argument("--duration", type=float, default=80)
    p.add_argument("--wavelengths", nargs="+", type=float, default=[0.45, 0.55, 0.65])
    p.add_argument("--case", choices=["pattern", "slab"], default="pattern")
    p.add_argument("--extinction", type=float, default=0.02)
    p.add_argument("--padding", type=float, default=0.8)
    p.add_argument("--pml", type=float, default=0.5)
    p.add_argument("--fwidth-factor", type=float, default=0.15)
    p.add_argument("--checkpoints", nargs="*", type=float, default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import meep as mp

    if args.slab_suite:
        rows = []
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for resolution in [80, 120, 160, 320, 640]:
            for extinction in [0.0, 0.005, 0.05]:
                for wavelength in args.wavelengths:
                    row = compute(
                        wavelength, resolution, 80.0, "slab", extinction, checkpoints=[20.0, 40.0]
                    )
                    row.update(resolution=resolution, extinction=extinction)
                    rows.append(row)
                    if mp.am_master():
                        atomic_result_write(args.output, {"case": "slab_suite", "spectra": rows})
        return
    result = collective_resume(mp, args.output, result_protocol(args))
    completed = len(result["spectra"])
    if result["status"] == "ok":
        if mp.am_master():
            print(f"Preserving completed checkpoint {args.output}", flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if mp.am_master():
        atomic_result_write(args.output, result)
        if completed:
            print(f"Resuming after {completed} completed wavelengths: {args.output}", flush=True)
    for wavelength in args.wavelengths[completed:]:
        row = compute(
            wavelength,
            args.resolution,
            args.duration,
            args.case,
            args.extinction,
            args.padding,
            args.pml,
            args.fwidth_factor,
            args.checkpoints,
            args.cw,
            args.adaptive_time,
        )
        result["spectra"].append(row)
        if mp.am_master():
            atomic_result_write(args.output, result)
            print(json.dumps(row), flush=True)
    result["status"] = "ok"
    if mp.am_master():
        atomic_result_write(args.output, result)


if __name__ == "__main__":
    main()
