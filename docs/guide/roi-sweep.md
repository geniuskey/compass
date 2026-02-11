---
title: ROI Sweep
description: Running a Region of Interest sweep across the sensor image plane in COMPASS, using CRA vs image height curves, microlens shift maps, and the ROISweepRunner.
---

# ROI Sweep

In a real camera, optical conditions vary across the sensor surface. Pixels at the sensor center see near-normal incidence (CRA close to zero), while pixels at the edges receive light at steep angles (CRA up to 30 degrees or more). The `ROISweepRunner` automates simulations at multiple sensor positions to predict spatially varying QE and relative illumination.

## How ROI sweep works

At each sensor position (defined by image height), the runner:

1. Interpolates the CRA from a user-provided CRA vs image height table
2. Applies the corresponding microlens shift to compensate for the oblique incidence
3. Sets the source angle to match the local CRA
4. Runs a full simulation at that position
5. Collects QE results for all positions into a unified output

## Basic usage

```python
from compass.runners.roi_sweep_runner import ROISweepRunner

config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "bayer_map": [["R", "G"], ["G", "B"]],
        "layers": {
            "microlens": {
                "enabled": True, "height": 0.6,
                "radius_x": 0.48, "radius_y": 0.48,
                "material": "polymer_n1p56",
                "shift": {"mode": "none"},
            },
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
            },
            "barl": {"layers": [
                {"thickness": 0.010, "material": "sio2"},
                {"thickness": 0.025, "material": "hfo2"},
            ]},
            "silicon": {
                "thickness": 3.0, "material": "silicon",
                "dti": {"enabled": True, "width": 0.1},
            },
        },
    },
    "solver": {
        "name": "torcwa", "type": "rcwa",
        "params": {"fourier_order": [9, 9]},
        "stability": {"precision_strategy": "mixed"},
    },
    "source": {
        "wavelength": {"mode": "single", "value": 0.55},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}

roi_config = {
    "image_heights": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "cra_table": [
        {"image_height": 0.0, "cra_deg": 0.0},
        {"image_height": 0.2, "cra_deg": 5.0},
        {"image_height": 0.4, "cra_deg": 10.0},
        {"image_height": 0.6, "cra_deg": 17.0},
        {"image_height": 0.8, "cra_deg": 24.0},
        {"image_height": 1.0, "cra_deg": 30.0},
    ],
}

results = ROISweepRunner.run(config, roi_config)
```

<ConeIlluminationViewer />

## Understanding the output

The output is a dictionary mapping position labels to `SimulationResult` objects:

```python
for key, result in results.items():
    avg_qe = sum(
        float(qe.mean()) for qe in result.qe_per_pixel.values()
    ) / len(result.qe_per_pixel)
    print(f"{key}: avg QE = {avg_qe:.3f}")

# Output:
# ih_0.00: avg QE = 0.712
# ih_0.20: avg QE = 0.698
# ih_0.40: avg QE = 0.671
# ih_0.60: avg QE = 0.623
# ih_0.80: avg QE = 0.558
# ih_1.00: avg QE = 0.481
```

## Defining the CRA table

The CRA table maps normalized image height (0.0 = center, 1.0 = corner) to the Chief Ray Angle in degrees. This curve depends on the camera lens design.

### From lens design data

If you have CRA data from Zemax, Code V, or another lens design tool:

```python
roi_config = {
    "image_heights": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "cra_table": [
        {"image_height": 0.0, "cra_deg": 0.0},
        {"image_height": 0.1, "cra_deg": 2.5},
        {"image_height": 0.2, "cra_deg": 5.1},
        {"image_height": 0.3, "cra_deg": 7.8},
        {"image_height": 0.4, "cra_deg": 10.8},
        {"image_height": 0.5, "cra_deg": 14.2},
        {"image_height": 0.6, "cra_deg": 17.5},
        {"image_height": 0.7, "cra_deg": 20.8},
        {"image_height": 0.8, "cra_deg": 24.0},
        {"image_height": 0.9, "cra_deg": 27.1},
        {"image_height": 1.0, "cra_deg": 30.0},
    ],
}
```

### Idealized linear CRA curve

For quick studies without lens data, a linear approximation is common:

```python
import numpy as np

max_cra_deg = 30.0
image_heights = np.linspace(0.0, 1.0, 11)
cra_table = [
    {"image_height": float(ih), "cra_deg": float(ih * max_cra_deg)}
    for ih in image_heights
]
```

### CRA interpolation

The runner uses `numpy.interp` for linear interpolation, so the CRA table does not need to match the `image_heights` sweep points exactly. You can provide a dense CRA table and sweep at coarser positions:

```python
roi_config = {
    "image_heights": [0.0, 0.5, 1.0],   # Only 3 positions
    "cra_table": cra_table,               # Dense 11-point table
}
```

## Microlens shift map

At each ROI position, the runner automatically sets the microlens shift mode to `auto_cra` and provides the interpolated CRA. The microlens is shifted laterally to re-center the focused light on the photodiode:

The shift direction and magnitude depend on the CRA. For a pixel at image height 0.6 with CRA = 17 degrees, the microlens is shifted toward the optical axis by approximately:

$$\Delta x \approx d \times \tan(\text{CRA})$$

where $d$ is the distance from the microlens to the photodiode.

This is handled automatically by the COMPASS geometry builder when `shift.mode = "auto_cra"`.

## Plotting QE vs image height

```python
import numpy as np
import matplotlib.pyplot as plt

image_heights = roi_config["image_heights"]
channel_colors = {"R": "red", "G": "green", "B": "blue"}

fig, ax = plt.subplots(figsize=(10, 6))

# Extract average QE per channel at each position
for channel in ["R", "G", "B"]:
    qe_vs_ih = []
    for ih in image_heights:
        key = f"ih_{ih:.2f}"
        result = results[key]
        channel_qe = [
            float(np.mean(qe))
            for name, qe in result.qe_per_pixel.items()
            if name.startswith(channel)
        ]
        qe_vs_ih.append(np.mean(channel_qe) if channel_qe else 0.0)
    ax.plot(image_heights, qe_vs_ih, "o-", color=channel_colors[channel],
            label=f"{channel} channel", linewidth=2)

ax.set_xlabel("Image Height (normalized)")
ax.set_ylabel("Average QE at 550 nm")
ax.set_title("QE vs Image Height (ROI Sweep)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roi_qe_vs_image_height.png", dpi=150)
```

## Relative illumination map

Relative illumination (RI) is the ratio of QE at each position to the QE at the center:

```python
center_key = "ih_0.00"
center_qe = np.mean([
    float(np.mean(qe)) for qe in results[center_key].qe_per_pixel.values()
])

ri_values = []
for ih in image_heights:
    key = f"ih_{ih:.2f}"
    pos_qe = np.mean([
        float(np.mean(qe)) for qe in results[key].qe_per_pixel.values()
    ])
    ri_values.append(pos_qe / center_qe)

plt.figure(figsize=(8, 5))
plt.plot(image_heights, ri_values, "o-", linewidth=2, color="navy")
plt.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("Image Height (normalized)")
plt.ylabel("Relative Illumination")
plt.title("Relative Illumination vs Image Height")
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("relative_illumination.png", dpi=150)
```

## Combining ROI sweep with wavelength sweep

For a full spectral ROI analysis, set the source to sweep mode:

```python
config["source"] = {
    "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.02}},
    "polarization": "unpolarized",
}

results = ROISweepRunner.run(config, roi_config)

# Now each result contains a full wavelength sweep
for key, result in results.items():
    print(f"{key}: {len(result.wavelengths)} wavelengths")
```

## Performance considerations

Each ROI position runs a complete simulation, so total runtime scales linearly with the number of image heights. For a 31-point wavelength sweep with 11 ROI positions:

- RCWA (torcwa, order [9,9]): ~110 s total (11 x 10 s per sweep)
- FDTD (flaport, 20 nm): ~500 s total (11 x 45 s per broadband run)

To reduce runtime, use fewer image height points for initial exploration and increase density only around regions of interest.

## Next steps

- [Cone Illumination](./cone-illumination.md) -- angular sampling for realistic illumination
- [CRA Shift Analysis cookbook](../cookbook/cra-shift-analysis.md) -- detailed CRA vs QE study
- [Visualization](./visualization.md) -- plotting tools for sweep results
