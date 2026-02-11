# DTI Crosstalk Analysis

This recipe demonstrates how to study the effect of Deep Trench Isolation (DTI) on optical crosstalk between adjacent pixels, including DTI depth and width sweeps.

## Background

DTI (Deep Trench Isolation) is a narrow trench etched between pixels and filled with a low-index material (typically SiO2). It acts as an optical waveguide barrier via total internal reflection, preventing photons generated in one pixel from diffusing into neighboring pixels. DTI is critical for small-pitch pixels (sub-1.2 um) where optical and electrical crosstalk would otherwise degrade color accuracy.

Key DTI parameters:

- **Width**: Typically 80--120 nm. Wider DTI isolates better but reduces photodiode area.
- **Depth**: Ranges from partial (1--2 um) to full-depth (equal to silicon thickness). Full-depth DTI provides the best isolation.
- **Material**: Usually SiO2 (n ~ 1.46). The refractive index contrast with silicon (n ~ 4.0) provides strong optical confinement.

## Setup

```python
import numpy as np
import copy
import matplotlib.pyplot as plt
from compass.runners.single_run import SingleRunner

base_config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "bayer_map": [["R", "G"], ["G", "B"]],
        "layers": {
            "air": {"thickness": 1.0, "material": "air"},
            "microlens": {
                "enabled": True, "height": 0.6,
                "radius_x": 0.48, "radius_y": 0.48,
                "material": "polymer_n1p56",
                "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
            },
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
            },
            "barl": {"layers": [
                {"thickness": 0.010, "material": "sio2"},
                {"thickness": 0.025, "material": "hfo2"},
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
        "params": {"fourier_order": [11, 11]},
        "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
    },
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01}},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}
```

Note: We use Fourier order [11, 11] instead of [9, 9] because the narrow DTI trench requires more harmonics to resolve accurately.

## With DTI vs without DTI

```python
# With DTI (baseline)
result_with = SingleRunner.run(base_config)

# Without DTI
config_no_dti = copy.deepcopy(base_config)
config_no_dti["pixel"]["layers"]["silicon"]["dti"]["enabled"] = False
result_without = SingleRunner.run(config_no_dti)
```

## Computing crosstalk

Crosstalk is the fraction of light intended for one pixel that is absorbed by a neighboring pixel. For a 2x2 Bayer unit cell, we measure how much green-channel light leaks into the red and blue pixels:

```python
def compute_crosstalk(result):
    """Compute crosstalk matrix from QE per pixel."""
    pixel_names = sorted(result.qe_per_pixel.keys())
    n_pixels = len(pixel_names)
    wavelengths = result.wavelengths

    # Average QE per pixel across all wavelengths
    qe_avg = {}
    for name in pixel_names:
        qe_avg[name] = float(np.mean(result.qe_per_pixel[name]))

    # Total QE across all pixels
    total = sum(qe_avg.values())

    # Crosstalk: fraction of total QE absorbed by each pixel
    crosstalk = {name: qe_avg[name] / total for name in pixel_names}
    return crosstalk, qe_avg

xt_with, qe_with = compute_crosstalk(result_with)
xt_without, qe_without = compute_crosstalk(result_without)

print("With DTI:")
for name, xt in xt_with.items():
    print(f"  {name}: QE={qe_with[name]:.3f}, fraction={xt:.3f}")

print("\nWithout DTI:")
for name, xt in xt_without.items():
    print(f"  {name}: QE={qe_without[name]:.3f}, fraction={xt:.3f}")
```

## DTI width sweep

Sweep the DTI trench width to find the optimal balance between isolation and fill factor:

```python
widths = [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
avg_qe_per_width = []
green_qe_per_width = []

for w in widths:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["silicon"]["dti"]["enabled"] = w > 0
    cfg["pixel"]["layers"]["silicon"]["dti"]["width"] = w
    result = SingleRunner.run(cfg)

    # Average QE across all pixels
    all_qe = np.mean([np.mean(qe) for qe in result.qe_per_pixel.values()])
    avg_qe_per_width.append(float(all_qe))

    # Green channel only
    green_qe = np.mean([
        np.mean(qe) for name, qe in result.qe_per_pixel.items()
        if name.startswith("G")
    ])
    green_qe_per_width.append(float(green_qe))

    print(f"DTI width={w*1000:.0f} nm: avg QE={all_qe:.3f}, green QE={green_qe:.3f}")
```

Plot the results:

```python
fig, ax = plt.subplots(figsize=(8, 5))

widths_nm = [w * 1000 for w in widths]
ax.plot(widths_nm, avg_qe_per_width, "o-", label="All channels avg", linewidth=2)
ax.plot(widths_nm, green_qe_per_width, "s-", label="Green channel avg", linewidth=2)
ax.set_xlabel("DTI Width (nm)")
ax.set_ylabel("Average QE")
ax.set_title("QE vs DTI Width")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dti_width_sweep.png", dpi=150)
```

## DTI depth sweep

For partial DTI, sweep the depth from 0 (no DTI) to full silicon thickness:

```python
si_thickness = 3.0
depths = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
qe_vs_depth = []

for d in depths:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["silicon"]["dti"]["enabled"] = d > 0
    cfg["pixel"]["layers"]["silicon"]["dti"]["depth"] = d
    cfg["pixel"]["layers"]["silicon"]["dti"]["width"] = 0.10
    result = SingleRunner.run(cfg)

    all_qe = np.mean([np.mean(qe) for qe in result.qe_per_pixel.values()])
    qe_vs_depth.append(float(all_qe))
    print(f"DTI depth={d:.1f} um: avg QE={all_qe:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(depths, qe_vs_depth, "o-", linewidth=2, color="tab:purple")
plt.xlabel("DTI Depth (um)")
plt.ylabel("Average QE")
plt.title(f"QE vs DTI Depth (Si thickness = {si_thickness} um)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dti_depth_sweep.png", dpi=150)
```

## Spectral crosstalk comparison

Plot wavelength-resolved QE for the red pixel with and without DTI to see where crosstalk is worst:

```python
fig, ax = plt.subplots(figsize=(10, 6))

wl_nm = result_with.wavelengths * 1000

# Red pixel QE with and without DTI
for name, qe in result_with.qe_per_pixel.items():
    if name.startswith("R"):
        ax.plot(wl_nm, qe, "r-", linewidth=2, label=f"{name} with DTI")
        break

for name, qe in result_without.qe_per_pixel.items():
    if name.startswith("R"):
        ax.plot(wl_nm, qe, "r--", linewidth=2, alpha=0.6, label=f"{name} no DTI")
        break

# Blue pixel QE with and without DTI
for name, qe in result_with.qe_per_pixel.items():
    if name.startswith("B"):
        ax.plot(wl_nm, qe, "b-", linewidth=2, label=f"{name} with DTI")
        break

for name, qe in result_without.qe_per_pixel.items():
    if name.startswith("B"):
        ax.plot(wl_nm, qe, "b--", linewidth=2, alpha=0.6, label=f"{name} no DTI")
        break

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("QE")
ax.set_title("Spectral QE: DTI Effect on Red and Blue Channels")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dti_spectral_crosstalk.png", dpi=150)
```

## Crosstalk heatmap

Visualize the cross-pixel energy distribution:

```python
from compass.visualization.qe_plot import plot_crosstalk_heatmap

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_crosstalk_heatmap(result_with, ax=ax1)
ax1.set_title("With DTI (100 nm)")

plot_crosstalk_heatmap(result_without, ax=ax2)
ax2.set_title("Without DTI")

plt.suptitle("Optical Crosstalk Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("dti_crosstalk_heatmap.png", dpi=150)
```

<CrosstalkHeatmap />

## Expected observations

1. **Without DTI**: Significant optical crosstalk, especially at long wavelengths (>600 nm) where photons penetrate deep into silicon and can diffuse laterally to neighboring pixels before being absorbed.
2. **With full-depth DTI**: Crosstalk is reduced by 5--10x. The SiO2 trench confines photons to the intended pixel via total internal reflection.
3. **Partial DTI**: Intermediate crosstalk reduction. Photons absorbed below the DTI depth can still cross over.
4. **DTI width trade-off**: Wider DTI improves isolation but reduces the active silicon area per pixel, slightly lowering peak QE. The sweet spot for 1 um pitch is typically 80--100 nm.
5. **Wavelength dependence**: Short wavelengths (blue, 400--500 nm) are absorbed near the surface and are less affected by DTI. Long wavelengths (red/NIR, 600--780 nm) benefit most from DTI because they penetrate deeper and have more opportunity for lateral diffusion.
