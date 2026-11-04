"""Current FDTDX API: periodic 3D cells, lossy media and reference-subtracted flux.

Units at this boundary are metres (FDTDX) and micrometres (COMPASS).
The template/JIT cache separates first-call compilation from repeated execution.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from compass.compute import cuda_device_index


@lru_cache(maxsize=16)
def _template(
    shape, spacing, wavelength, duration, pml_cells, device, polarization, courant_factor, precision
):
    import fdtdx as fx
    import jax
    import jax.numpy as jnp

    if precision != "float32":
        jax.config.update("jax_enable_x64", True)
    target = jax.devices("gpu" if device.startswith("cuda") else "cpu")[cuda_device_index(device)]
    jax.config.update("jax_default_device", target)
    config = fx.SimulationConfig(
        time=duration * 1e-6 / 299792458,
        grid=fx.UniformGrid(spacing=spacing * 1e-6),
        backend=target.platform,
        dtype=jnp.float32 if precision == "float32" else jnp.float64,
        courant_factor=courant_factor,
    )
    volume = fx.SimulationVolume(name="volume", partial_grid_shape=shape)
    boundaries, constraints = fx.boundary_objects_from_config(
        fx.BoundaryConfig(
            boundary_type_minx="periodic",
            boundary_type_maxx="periodic",
            boundary_type_miny="periodic",
            boundary_type_maxy="periodic",
            thickness_grid_minz=pml_cells,
            thickness_grid_maxz=pml_cells,
        ),
        volume,
    )
    wave = fx.WaveCharacter(wavelength=wavelength * 1e-6)
    source_z = shape[2] // 2 - pml_cells - max(2, round(0.2 / spacing))
    # Physical positions are relative to the volume centre, in metres.
    source = fx.UniformPlaneSource(
        name="source",
        partial_grid_shape=(shape[0], shape[1], 1),
        partial_real_position=(0.0, 0.0, source_z * spacing * 1e-6),
        direction="-",
        wave_character=wave,
        fixed_E_polarization_vector=(0.0, 1.0, 0.0) if polarization == "TE" else (1.0, 0.0, 0.0),
        temporal_profile=fx.GaussianPulseProfile(
            center_wave=wave, spectral_width=fx.WaveCharacter(frequency=0.15 * wave.get_frequency())
        ),
    )
    monitors = []
    for name, z in [
        ("reflection", source_z + max(2, round(0.1 / spacing))),
        ("transmission", -source_z + max(2, round(0.1 / spacing))),
    ]:
        monitors.append(
            fx.PhasorPoyntingFluxDetector(
                name=name,
                partial_grid_shape=(shape[0], shape[1], 1),
                partial_real_position=(0.0, 0.0, z * spacing * 1e-6),
                direction="+",
                wave_characters=[wave],
                dtype=jnp.complex64 if precision == "float32" else jnp.complex128,
                scaling_mode="pulse",
                dft_subsample=1,
            )
        )
    objects, arrays, _, config, _ = fx.place_objects(
        [volume, *boundaries.values(), source, *monitors], config, constraints
    )
    # Always allocate conductivity, including the empty reference; no loss gets dropped.
    arrays = arrays.aset("electric_conductivity", jnp.zeros((1, *shape), dtype=config.dtype))
    run = jax.jit(lambda a: fx.run_fdtd(a, objects, config, show_progress=False)[1])
    detectors = {d.name: d for d in objects.detectors}
    return arrays, run, detectors, target


def solve(
    epsilon_yxz,
    wavelength,
    spacing,
    duration,
    pml_cells,
    device,
    polarization,
    *,
    courant_factor=0.5,
    precision="float64",
):
    """Return raw R/T/A and observed device; no clipping or QE surrogate."""
    import jax
    import jax.numpy as jnp

    epsilon = np.asarray(epsilon_yxz).transpose(1, 0, 2)
    if np.any(epsilon.real <= 0):
        raise NotImplementedError("FDTDX voxel adapter requires positive real permittivity")
    buffer = max(2, round(0.8 / spacing))
    pad = buffer + pml_cells
    epsilon = np.pad(epsilon, ((0, 0), (0, 0), (pad, pad)), constant_values=1)
    arrays, run, detectors, target = _template(
        epsilon.shape,
        spacing,
        wavelength,
        duration,
        pml_cells,
        device,
        polarization,
        courant_factor,
        precision,
    )
    with jax.default_device(target):
        reference = run(arrays)
        material = arrays.aset(
            "inv_permittivities",
            jnp.asarray(1 / epsilon.real, dtype=arrays.inv_permittivities.dtype)[None, ...],
        )
        sigma = 2 * np.pi * 299792458 / (wavelength * 1e-6) * 8.8541878128e-12 * epsilon.imag
        material = material.aset(
            "electric_conductivity",
            jnp.asarray(sigma * spacing * 1e-6, dtype=arrays.inv_permittivities.dtype)[None, ...],
        )
        result = run(material)
        tr = detectors["transmission"]
        rr = detectors["reflection"]
        incident = float(tr.compute_poynting_flux(reference.detector_states[tr.name])[0])
        reflected_state = {
            "phasor": result.detector_states[rr.name]["phasor"]
            - reference.detector_states[rr.name]["phasor"]
        }
        reflected = float(rr.compute_poynting_flux(reflected_state)[0])
        transmitted = float(tr.compute_poynting_flux(result.detector_states[tr.name])[0])
    if not np.isfinite(incident) or incident >= -1e-30:
        raise RuntimeError(f"Invalid FDTDX incident signed flux {incident}")
    R, T = -reflected / incident, transmitted / incident
    return R, T, 1 - R - T, str(result.fields.E.device)
