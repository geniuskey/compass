# Wavelength Sweep

This recipe demonstrates a full visible-range wavelength sweep and how to analyze the spectral data.

## Basic sweep

The simplest wavelength sweep uses the `"sweep"` mode in the source config:

```python
from compass.runners.single_run import SingleRunner

config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "bayer_map": [["R", "G"], ["G", "B"]],
        "layers": {
            "microlens": {"enabled": True, "height": 0.6, "radius_x": 0.48, "radius_y": 0.48},
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
            },
            "barl": {"layers": [
                {"thickness": 0.010, "material": "sio2"},
                {"thickness": 0.025, "material": "hfo2"},
                {"thickness": 0.015, "material": "sio2"},
                {"thickness": 0.030, "material": "si3n4"},
            ]},
            "silicon": {
                "thickness": 3.0, "material": "silicon",
                "photodiode": {"position": [0, 0, 0.5], "size": [0.7, 0.7, 2.0]},
                "dti": {"enabled": True, "width": 0.1, "material": "sio2"},
            },
        },
    },
    "solver": {
        "name": "torcwa", "type": "rcwa",
        "params": {"fourier_order": [9, 9]},
        "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
    },
    "source": {
        "wavelength": {
            "mode": "sweep",
            "sweep": {"start": 0.38, "stop": 0.78, "step": 0.01},
        },
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}

result = SingleRunner.run(config)
print(f"Sweep: {len(result.wavelengths)} wavelengths from "
      f"{result.wavelengths[0]*1000:.0f} to {result.wavelengths[-1]*1000:.0f} nm")
```

## Full visible spectrum plot

```python
import matplotlib.pyplot as plt
import numpy as np
from compass.visualization.qe_plot import plot_qe_spectrum

fig, ax = plt.subplots(figsize=(10, 6))
plot_qe_spectrum(result, ax=ax)

# Add wavelength color bar on x-axis for reference
wl_nm = result.wavelengths * 1000
for i in range(len(wl_nm) - 1):
    color = wavelength_to_rgb(wl_nm[i])  # You'd need a helper for this
    ax.axvspan(wl_nm[i], wl_nm[i+1], alpha=0.03, color=color)

ax.set_title("Full Visible Spectrum QE")
plt.tight_layout()
plt.savefig("full_spectrum_qe.png", dpi=200)
```

## Reflectance, transmittance, absorption

Plot the energy balance components alongside QE:

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# QE
plot_qe_spectrum(result, ax=ax1)
ax1.set_xlabel("")

# R, T, A
wl_nm = result.wavelengths * 1000
if result.reflection is not None:
    ax2.plot(wl_nm, result.reflection, label="Reflection", color="tab:blue")
if result.transmission is not None:
    ax2.plot(wl_nm, result.transmission, label="Transmission", color="tab:orange")
if result.absorption is not None:
    ax2.plot(wl_nm, result.absorption, label="Absorption", color="tab:red")

ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Fraction")
ax2.set_title("Energy Balance Components")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("qe_and_energy.png", dpi=150)
```

## Fine-resolution sweep

For resolving thin-film interference fringes, use a finer step:

```python
config_fine = config.copy()
config_fine["source"] = {
    "wavelength": {
        "mode": "sweep",
        "sweep": {"start": 0.50, "stop": 0.60, "step": 0.002},  # 2 nm step
    },
    "polarization": "unpolarized",
}

result_fine = SingleRunner.run(config_fine)
```

The 2 nm step resolves the Fabry-Perot fringes from the silicon cavity (fringe spacing ~12 nm for 3 um Si at 550 nm).

## Wavelength list mode

For specific wavelengths (e.g., laser lines or LED peaks):

```python
config_list = config.copy()
config_list["source"] = {
    "wavelength": {
        "mode": "list",
        "values": [0.405, 0.450, 0.525, 0.590, 0.625, 0.680, 0.780],
    },
    "polarization": "unpolarized",
}

result_list = SingleRunner.run(config_list)
```

## Comparing different silicon thicknesses

Run sweeps for different silicon thicknesses and overlay:

```python
import copy

thicknesses = [2.0, 3.0, 4.0]
results_by_thickness = []

for t in thicknesses:
    cfg = copy.deepcopy(config)
    cfg["pixel"]["layers"]["silicon"]["thickness"] = t
    cfg["pixel"]["layers"]["silicon"]["dti"]["depth"] = t
    r = SingleRunner.run(cfg)
    results_by_thickness.append(r)

from compass.visualization.qe_plot import plot_qe_comparison

fig, ax = plt.subplots(figsize=(10, 6))
plot_qe_comparison(
    results=results_by_thickness,
    labels=[f"Si {t}um" for t in thicknesses],
    ax=ax,
)
ax.set_title("QE vs Silicon Thickness")
plt.tight_layout()
plt.savefig("si_thickness_comparison.png", dpi=150)
```

## Key observations

| Wavelength range | Observation |
|-----------------|-------------|
| 380-420 nm | Low QE due to surface recombination and shallow absorption |
| 420-500 nm | Blue channel peak. Short absorption depth, sensitive to BARL design |
| 500-580 nm | Green channel peak. Moderate absorption depth |
| 580-650 nm | Red channel peak. Longer absorption depth, benefits from thicker Si |
| 650-780 nm | QE drops as absorption depth exceeds silicon thickness |

::: tip
For NIR (near-infrared) applications, increase silicon thickness to 5-6 um and extend the sweep to 1000 nm. You may need to add silicon material data beyond 780 nm.
:::
