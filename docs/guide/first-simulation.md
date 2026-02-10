# First Simulation

This guide walks through a complete COMPASS simulation from start to finish: loading a pixel config, running a wavelength sweep with the torcwa RCWA solver, computing QE, and saving results.

## Overview

We will simulate a 2x2 BSI (back-side illuminated) pixel unit cell with a 1.0 um pitch, sweep wavelengths from 400 to 700 nm at normal incidence, and plot the resulting QE spectrum per color channel.

## Step 1: Load configuration

```python
from pathlib import Path
from compass.core.config_schema import CompassConfig
from omegaconf import OmegaConf

# Load from YAML via Hydra/OmegaConf
yaml_path = Path("configs/pixel/default_bsi_1um.yaml")
raw = OmegaConf.load(yaml_path)

config = CompassConfig(**{
    "pixel": OmegaConf.to_container(raw["pixel"], resolve=True),
    "solver": {
        "name": "torcwa",
        "type": "rcwa",
        "params": {"fourier_order": [9, 9], "dtype": "complex64"},
    },
    "source": {
        "type": "planewave",
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01}},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
})
```

## Step 2: Build geometry

```python
from compass.materials.database import MaterialDB
from compass.geometry.builder import GeometryBuilder

mat_db = MaterialDB()
builder = GeometryBuilder(config.pixel, mat_db)
pixel_stack = builder.build()

print(f"Pixel: {config.pixel.pitch} um pitch, "
      f"{config.pixel.unit_cell[0]}x{config.pixel.unit_cell[1]} unit cell")
print(f"Stack: {pixel_stack.total_thickness:.2f} um total, "
      f"{len(pixel_stack.layers)} layers")
```

## Step 3: Verify the structure

Always visually check the geometry before running a solver.

```python
from compass.visualization.structure_plot_2d import plot_pixel_cross_section

# XZ cross-section through pixel center
ax = plot_pixel_cross_section(
    pixel_stack,
    plane="xz",
    position=config.pixel.pitch / 2,
    wavelength=0.55,
    figsize=(12, 6),
)
```

Check that:
- Microlenses sit on top of the planarization layer
- Color filter shows the expected Bayer pattern colors
- DTI trenches are visible in the silicon layer
- Layer thicknesses look reasonable

## Step 4: Create the solver

```python
from compass.solvers.base import SolverFactory

solver = SolverFactory.create("torcwa", config.solver)
print(f"Solver: {solver.name}, device: {solver.device}")
```

## Step 5: Run wavelength sweep

```python
import numpy as np
from compass.sources.planewave import PlaneWave

wavelengths = np.arange(0.40, 0.701, 0.01)  # 400-700 nm in um
results = []

for wl in wavelengths:
    source_cfg = {
        "wavelength": wl,
        "theta": 0.0,
        "phi": 0.0,
        "polarization": "unpolarized",
    }

    solver.setup_geometry(pixel_stack)
    solver.setup_source(source_cfg)
    result = solver.run_timed()
    results.append(result)

    # Check energy balance
    is_valid = solver.validate_energy_balance(result, tolerance=0.01)
    if not is_valid:
        print(f"  WARNING: energy violation at {wl*1000:.0f} nm")

print(f"Completed {len(results)} wavelength points")
```

## Step 6: Extract QE per pixel

```python
qe_data = {}  # pixel_name -> qe_array

for pixel_name in results[0].qe_per_pixel.keys():
    qe_spectrum = np.array([r.qe_per_pixel[pixel_name] for r in results])
    qe_data[pixel_name] = qe_spectrum.flatten()

print("Pixels found:", list(qe_data.keys()))
# Example: ['R_0_0', 'G_0_1', 'G_1_0', 'B_1_1']
```

## Step 7: Plot QE spectrum

```python
import matplotlib.pyplot as plt

wavelengths_nm = wavelengths * 1000  # Convert um to nm

fig, ax = plt.subplots(figsize=(10, 6))

channel_colors = {"R": "red", "G": "green", "B": "blue"}
channel_qe = {"R": [], "G": [], "B": []}

for pixel_name, qe in qe_data.items():
    channel = pixel_name[0]  # First character: R, G, or B
    channel_qe[channel].append(qe)
    ax.plot(wavelengths_nm, qe, color=channel_colors[channel],
            alpha=0.4, linewidth=0.8)

# Plot channel averages
for channel, qe_list in channel_qe.items():
    if qe_list:
        mean_qe = np.mean(qe_list, axis=0)
        ax.plot(wavelengths_nm, mean_qe, color=channel_colors[channel],
                linewidth=2.5, label=f"{channel} (mean)")

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Quantum Efficiency")
ax.set_title("BSI 1.0um Pixel QE Spectrum (torcwa, normal incidence)")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("qe_spectrum.png", dpi=150)
```

## Step 8: Check energy balance

```python
R_values = np.array([r.reflection.mean() if r.reflection is not None else 0
                      for r in results])
T_values = np.array([r.transmission.mean() if r.transmission is not None else 0
                      for r in results])
A_values = np.array([r.absorption.mean() if r.absorption is not None else 0
                      for r in results])

energy_sum = R_values + T_values + A_values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

ax1.plot(wavelengths_nm, R_values, label="R (reflectance)", color="tab:blue")
ax1.plot(wavelengths_nm, T_values, label="T (transmittance)", color="tab:orange")
ax1.plot(wavelengths_nm, A_values, label="A (absorption)", color="tab:red")
ax1.set_ylabel("Fraction")
ax1.set_title("Energy Balance: R + T + A")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(wavelengths_nm, energy_sum - 1.0, color="black")
ax2.axhline(0, color="gray", linestyle="--")
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("R+T+A - 1")
ax2.set_title("Energy Conservation Error")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
```

The error should be below 1% (0.01) at all wavelengths. If it exceeds this, see [Troubleshooting](./troubleshooting.md).

## Step 9: Save results

```python
from compass.io.hdf5_handler import save_results_hdf5

save_results_hdf5(
    results,
    wavelengths=wavelengths,
    pixel_stack=pixel_stack,
    filepath="outputs/bsi_1um_torcwa_qe.h5",
    metadata={
        "solver": "torcwa",
        "fourier_order": [9, 9],
        "polarization": "unpolarized",
        "angle_theta": 0.0,
    },
)
print("Results saved to outputs/bsi_1um_torcwa_qe.h5")
```

## Using the script runner

For production runs, use the command-line runner which handles config loading, sweeps, and output automatically:

```bash
# Default simulation
python scripts/run_simulation.py

# Override solver and pixel config
python scripts/run_simulation.py solver=torcwa pixel=default_bsi_1um

# Wavelength sweep
python scripts/run_simulation.py \
    source.wavelength.mode=sweep \
    source.wavelength.sweep.start=0.40 \
    source.wavelength.sweep.stop=0.70 \
    source.wavelength.sweep.step=0.01
```

## Next steps

- [Pixel Stack Config](./pixel-stack-config.md) -- configure different pixel structures
- [Material Database](./material-database.md) -- add custom materials
- [Choosing a Solver](./choosing-solver.md) -- compare solver options
