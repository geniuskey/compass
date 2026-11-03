"""Native fdtdz lossless slab test, isolated from the periodic lossy benchmark.

fdtdz has no material absorption, CPU kernel, or periodic-xy boundary API.
A wide uniform current sheet is observed at its centre before transverse
boundary signals arrive. Width/time convergence checks the normal-plane limit.
R/T use reference-subtracted electric amplitudes in homogeneous air; this is
valid for this uniform normal-incidence slab, not a diffraction-power extractor.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def arrays(nxy, dx, wavelength, time_limit):
    import jax.numpy as jnp

    nz, dt, pml = 48, 0.5, 8
    steps = round(time_limit / (dx * dt))
    t = np.arange(steps) * dt
    frequency = dx / wavelength
    sigma = 1 / (2 * np.pi * 0.6 * frequency)
    pulse = np.exp(-0.5 * ((t - 6 * sigma) / sigma) ** 2) * np.sin(
        2 * np.pi * frequency * (t - 6 * sigma)
    )
    waveform = np.stack([pulse, np.zeros_like(pulse)], axis=1).astype(np.float32)
    source = np.zeros((2, 2, nxy, nxy, 1), np.float32)
    source[0, 1, :, :, 0] = 1
    boundary = 48
    x = np.arange(nxy)
    u = np.maximum(np.maximum(boundary - x, x - (nxy - 1 - boundary)), 0) / boundary
    mask = 0.25 * (u[:, None] ** 4 + u[None, :] ** 4)
    absorption = np.broadcast_to(mask, (3, nxy, nxy)).astype(np.float32).copy()
    sigma_z = np.zeros((nz, 2), np.float32)
    kappa = np.ones((nz, 2), np.float32)
    alpha = np.zeros((nz, 2), np.float32)
    for shift in [0, 1]:
        z = np.arange(nz) + 0.5 * shift
        depth = np.clip(np.maximum(pml - z, z - (nz - pml)) / pml, 0, 1)
        inside = depth > 0
        sigma_z[:, shift] = 4.6 * depth**3
        kappa[:, shift] = 1 + 4 * depth**3
        alpha[:, shift] = np.where(inside, 0.05 * (1 - depth), 0)
    reference = np.ones((3, 2, 2, nz), np.float32)
    structure = reference.copy()
    # Centre an exactly 0.1-um slab on Ey nodes (tangential to z).
    thickness = round(0.1 / dx)
    lower = nz // 2 - thickness // 2
    structure[:, :, :, lower : lower + thickness] = 1.5**2
    arguments = dict(
        dt=dt,
        source_field=jnp.asarray(source),
        source_waveform=jnp.asarray(waveform),
        source_position=36,
        absorption_mask=jnp.asarray(absorption),
        pml_kappa=jnp.asarray(kappa),
        pml_sigma=jnp.asarray(sigma_z),
        pml_alpha=jnp.asarray(alpha),
        pml_widths=(pml, pml),
        output_steps=(0, steps, 1),
        use_reduced_precision=False,
        launch_params=((2, 4), (4, 4), 2, (8, 0)),
        offset=(nxy // 2 - 1, nxy // 2 - 1, 0),
    )
    return jnp.asarray(reference), jnp.asarray(structure), arguments


def measure(reference, structure, arguments, wavelength, dx):
    from fdtdz_jax import fdtdz

    ref = fdtdz(epsilon=reference, **arguments)
    ref.block_until_ready()
    actual = fdtdz(epsilon=structure, **arguments)
    actual.block_until_ready()
    device = actual.device
    observed_device = str(device() if callable(device) else device)
    ref = np.asarray(ref).mean(axis=(2, 3))
    actual = np.asarray(actual).mean(axis=(2, 3))
    steps = ref.shape[0]
    phase = np.exp(2j * np.pi * dx / wavelength * np.arange(steps) * arguments["dt"])
    e_ref = np.einsum("t,tcz->cz", phase, ref)
    e_actual = np.einsum("t,tcz->cz", phase, actual)
    # Reflection plane lies between source (36) and slab; transmission below it.
    reflection_plane, transmission_plane = 31, 15
    R = (
        abs(
            (e_actual[1, reflection_plane] - e_ref[1, reflection_plane])
            / e_ref[1, reflection_plane]
        )
        ** 2
    )
    T = abs(e_actual[1, transmission_plane] / e_ref[1, transmission_plane]) ** 2
    if not np.isfinite([R, T, 1 - R - T]).all():
        raise RuntimeError("Non-finite native fdtdz flux ratio")
    return [float(R), float(T), float(1 - R - T)], str(ref.dtype), observed_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "solver": "fdtdz_native",
        "case": "lossless_slab_native",
        "status": "failed",
        "python": sys.version,
        "spectra": [],
        "launch_params": [[2, 4], [4, 4], 2, [8, 0]],
        "precision": "float32",
        "timing_context": "concurrent_validation",
        "timing_includes": "material/source allocation, reference and structure kernels, synchronization, host transfers and DFT reduction",
        "timing_excludes": "Python/JAX startup and initial GPU capability probe",
        "timing_protocol": {"cold_repeats": 1, "warm_repeats": 3},
        "pml_widths": [8, 8],
        "xy_absorber_width_cells": 48,
        "slab_index": 1.5,
        "slab_thickness_um": 0.1,
        "dt_cell_units": 0.5,
        "memory_measurement": "not reported: native workspace is outside the JAX allocator",
        "scope": "uniform lossless slab; causal central-plane limit; not periodic patterned-cell R/T",
    }
    try:
        import jax
        import jax.numpy as jnp
        import jaxlib

        result["jax"] = jax.__version__
        result["jaxlib"] = jaxlib.__version__
        gpu = jax.devices("gpu")[0]
        result["device"] = str(gpu)
        result["gpu"] = gpu.device_kind
        probe = jax.device_put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), gpu)
        (probe @ probe.T).block_until_ready()
        result["gpu_probe_device"] = str(probe.device)
        result["status"] = "running"
        for dx, nxy, limit in [
            (0.025, 384, 3.0),
            (0.025, 384, 3.5),
            (0.025, 512, 3.5),
            (0.0125, 768, 3.5),
        ]:
            case = {
                "grid_spacing": dx,
                "transverse_cells": nxy,
                "duration_um_over_c": limit,
                "wavelength_um": [0.45, 0.55, 0.65],
                "rta": [],
                "cold_s": None,
                "warm_s": [],
            }
            outputs = []
            timings = []
            for repeat in range(4):
                start = time.perf_counter()
                spectrum = []
                for wavelength in case["wavelength_um"]:
                    # Material/source allocation is included equally in every repeat.
                    ref, structure, kw = arrays(nxy, dx, wavelength, limit)
                    actual, dtype, observed = measure(ref, structure, kw, wavelength, dx)
                    case["actual_device"] = observed
                    case["dtype"] = dtype
                    spectrum.append(actual)
                timings.append(time.perf_counter() - start)
                outputs.append(spectrum)
                print(dx, nxy, limit, repeat, timings[-1], spectrum, flush=True)
            case.update(
                rta=outputs[-1],
                cold_s=timings[0],
                warm_s=timings[1:],
                warm_median_s=statistics.median(timings[1:]),
                max_repeat_delta=float(np.max(np.abs(np.array(outputs) - outputs[0]))),
            )
            result["spectra"].append(case)
            args.output.write_text(json.dumps(result, indent=2))
        # Exercise CPU placement explicitly: this primitive has only a CUDA lowering.
        try:
            from fdtdz_jax import fdtdz

            with jax.default_device(jax.devices("cpu")[0]):
                ref, structure, kw = arrays(128, 0.025, 0.55, 1.0)
                fdtdz(epsilon=ref, **kw).block_until_ready()
            result["cpu_status"] = "unexpected_cpu_execution"
        except Exception as exc:
            result["cpu_status"] = "unsupported"
            result["cpu_error"] = str(exc)
        result["status"] = "ok"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
