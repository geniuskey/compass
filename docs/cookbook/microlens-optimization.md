# Microlens Optimization

This recipe shows how to sweep microlens parameters (height, radius, squareness) to maximize QE and study the effect of CRA (Chief Ray Angle).

## Background

The microlens is the most optically significant element in a BSI pixel. Its shape determines how well light is focused onto the photodiode. Key parameters:

- **Height** ($H$): Controls the focusing power. Too tall and light over-focuses; too short and it under-focuses.
- **Radius** ($r_x, r_y$): Determines the lens aperture. Should be close to but less than half the pitch.
- **Squareness** ($n$): Superellipse parameter. $n=2$ is an ellipse; higher values create a more box-like lens with better fill factor.
- **CRA shift**: For pixels at the edge of a sensor, the microlens must be shifted to accept light at an angle.

## Setup

```python
import numpy as np
import copy
import matplotlib.pyplot as plt
from compass.runners.single_run import SingleRunner
from compass.analysis.qe_calculator import QECalculator

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
        "wavelength": {"mode": "single", "value": 0.55},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}
```

<StaircaseMicrolensViewer />

## Sweep 1: Microlens height

```python
heights = np.arange(0.2, 1.01, 0.1)
avg_qe_vs_height = []

for h in heights:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["microlens"]["height"] = float(h)
    result = SingleRunner.run(cfg)

    # Average QE across all pixels
    all_qe = np.mean([qe[0] for qe in result.qe_per_pixel.values()])
    avg_qe_vs_height.append(all_qe)
    print(f"  height={h:.1f} um -> avg QE = {all_qe:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(heights, avg_qe_vs_height, "o-", linewidth=2)
plt.xlabel("Microlens Height (um)")
plt.ylabel("Average QE at 550 nm")
plt.title("QE vs Microlens Height")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ml_height_sweep.png", dpi=150)
```

## Sweep 2: Squareness parameter

```python
n_values = [2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
avg_qe_vs_n = []

for n in n_values:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["microlens"]["profile"]["n"] = n
    result = SingleRunner.run(cfg)

    all_qe = np.mean([qe[0] for qe in result.qe_per_pixel.values()])
    avg_qe_vs_n.append(all_qe)
    print(f"  n={n:.1f} -> avg QE = {all_qe:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(n_values, avg_qe_vs_n, "s-", linewidth=2)
plt.xlabel("Superellipse Squareness (n)")
plt.ylabel("Average QE at 550 nm")
plt.title("QE vs Microlens Squareness")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ml_squareness_sweep.png", dpi=150)
```

## Sweep 3: CRA response

Study how QE degrades as the chief ray angle increases:

```python
cra_angles = [0, 5, 10, 15, 20, 25, 30]
results_no_shift = []
results_with_shift = []

for cra in cra_angles:
    # Without microlens shift
    cfg = copy.deepcopy(base_config)
    cfg["source"]["angle"] = {"theta_deg": float(cra), "phi_deg": 0.0}
    cfg["pixel"]["layers"]["microlens"]["shift"]["mode"] = "none"
    r = SingleRunner.run(cfg)
    results_no_shift.append(r)

    # With auto CRA shift
    cfg2 = copy.deepcopy(cfg)
    cfg2["pixel"]["layers"]["microlens"]["shift"]["mode"] = "auto_cra"
    cfg2["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = float(cra)
    r2 = SingleRunner.run(cfg2)
    results_with_shift.append(r2)
    print(f"  CRA={cra}deg: done")

# Plot
qe_no_shift = [np.mean([qe[0] for qe in r.qe_per_pixel.values()])
                for r in results_no_shift]
qe_with_shift = [np.mean([qe[0] for qe in r.qe_per_pixel.values()])
                  for r in results_with_shift]

plt.figure(figsize=(8, 5))
plt.plot(cra_angles, qe_no_shift, "o-", label="No ML shift", linewidth=2)
plt.plot(cra_angles, qe_with_shift, "s-", label="Auto CRA shift", linewidth=2)
plt.xlabel("Chief Ray Angle (degrees)")
plt.ylabel("Average QE at 550 nm")
plt.title("QE vs CRA: Effect of Microlens Shift")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ml_cra_sweep.png", dpi=150)
```

## Expected results

- **Height**: QE peaks at an optimal height (typically 0.5-0.7 um for 1 um pitch). Too tall causes over-focusing; too short gives weak focusing.
- **Squareness**: Higher $n$ values (more square lens) generally improve QE because of better fill factor, but very high values can cause edge diffraction.
- **CRA**: QE drops at high CRA without microlens shift. With auto CRA shift, the lens compensates and maintains QE up to 15-20 degrees before declining.

## 2D optimization

For a more thorough study, sweep height and radius simultaneously:

```python
heights = np.arange(0.3, 0.9, 0.1)
radii = np.arange(0.35, 0.50, 0.02)
qe_map = np.zeros((len(heights), len(radii)))

for i, h in enumerate(heights):
    for j, r in enumerate(radii):
        cfg = copy.deepcopy(base_config)
        cfg["pixel"]["layers"]["microlens"]["height"] = float(h)
        cfg["pixel"]["layers"]["microlens"]["radius_x"] = float(r)
        cfg["pixel"]["layers"]["microlens"]["radius_y"] = float(r)
        result = SingleRunner.run(cfg)
        qe_map[i, j] = np.mean([qe[0] for qe in result.qe_per_pixel.values()])

plt.figure(figsize=(8, 6))
plt.pcolormesh(radii, heights, qe_map, shading="auto", cmap="viridis")
plt.colorbar(label="Average QE")
plt.xlabel("Microlens Radius (um)")
plt.ylabel("Microlens Height (um)")
plt.title("QE vs Height and Radius")
plt.tight_layout()
plt.savefig("ml_2d_optimization.png", dpi=150)
```
