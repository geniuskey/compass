# Metal Grid Effect

This recipe demonstrates how the tungsten metal grid between color filter sub-pixels affects QE and optical crosstalk.

## Background

The metal grid sits between adjacent color filter elements. It provides optical isolation by blocking light from entering neighboring pixels through the color filter layer. However, it also:

- Reduces the effective aperture (less light enters each pixel)
- Can cause diffraction effects at the grid edges
- Absorbs some light (tungsten is lossy)

This recipe runs two simulations -- with and without the metal grid -- and compares the results.

## Setup

```python
import copy
from compass.runners.single_run import SingleRunner
from compass.analysis.solver_comparison import SolverComparison
from compass.visualization.qe_plot import plot_qe_comparison, plot_crosstalk_heatmap
import matplotlib.pyplot as plt

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
                "shift": {"mode": "none"},
            },
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "pattern": "bayer_rggb",
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
            },
            "barl": {
                "layers": [
                    {"thickness": 0.010, "material": "sio2"},
                    {"thickness": 0.025, "material": "hfo2"},
                ]
            },
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

## Run: with metal grid

```python
config_with_grid = copy.deepcopy(base_config)
config_with_grid["pixel"]["layers"]["color_filter"]["grid"]["enabled"] = True

result_with = SingleRunner.run(config_with_grid)
print("With grid: done")
```

## Run: without metal grid

```python
config_no_grid = copy.deepcopy(base_config)
config_no_grid["pixel"]["layers"]["color_filter"]["grid"]["enabled"] = False

result_without = SingleRunner.run(config_no_grid)
print("Without grid: done")
```

## Compare QE spectra

```python
fig, ax = plt.subplots(figsize=(10, 6))
plot_qe_comparison(
    results=[result_with, result_without],
    labels=["With grid", "No grid"],
    ax=ax,
)
ax.set_title("Metal Grid Effect on QE")
plt.tight_layout()
plt.savefig("metal_grid_qe_comparison.png", dpi=150)
plt.show()
```

## Quantify the difference

```python
comparison = SolverComparison(
    results=[result_with, result_without],
    labels=["with_grid", "no_grid"],
    reference_idx=0,
)
summary = comparison.summary()

for key, val in summary["max_qe_diff"].items():
    print(f"{key}: max |dQE| = {val:.4f}")
```

## Compare crosstalk

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_crosstalk_heatmap(result_with, ax=ax1)
ax1.set_title("Crosstalk: With Grid")

plot_crosstalk_heatmap(result_without, ax=ax2)
ax2.set_title("Crosstalk: No Grid")

plt.tight_layout()
plt.savefig("metal_grid_crosstalk.png", dpi=150)
plt.show()
```

## Expected observations

1. **QE reduction with grid**: The metal grid slightly reduces peak QE (typically 2-5%) because it blocks some light and absorbs energy.
2. **Crosstalk improvement**: The grid significantly reduces optical crosstalk between adjacent pixels, especially for off-axis illumination.
3. **Wavelength dependence**: The grid effect is stronger at shorter wavelengths where diffraction effects are more pronounced relative to the grid width.

## Grid width sweep

Study how grid width affects the QE/crosstalk trade-off:

```python
import numpy as np

grid_widths = [0.0, 0.03, 0.05, 0.08, 0.10]
results_vs_width = []

for width in grid_widths:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["color_filter"]["grid"]["enabled"] = width > 0
    cfg["pixel"]["layers"]["color_filter"]["grid"]["width"] = width
    r = SingleRunner.run(cfg)
    results_vs_width.append(r)
    print(f"Grid width {width*1000:.0f} nm: done")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_qe_comparison(
    results=results_vs_width,
    labels=[f"w={w*1000:.0f}nm" for w in grid_widths],
    ax=ax,
)
ax.set_title("QE vs Metal Grid Width")
plt.tight_layout()
plt.savefig("grid_width_sweep.png", dpi=150)
```
