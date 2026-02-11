# BARL Design

This recipe demonstrates multi-layer Bottom Anti-Reflective Layer (BARL) optimization, including the quarter-wave condition, thickness sweeps, and broadband ARC design for BSI pixels.

## Background

The BARL sits between the color filter and the silicon photodiode. Without an anti-reflective coating, the air/SiO2/silicon interface reflects 15--30% of incident light, significantly reducing QE. A well-designed BARL stack reduces this reflection to under 5% across the visible spectrum.

The quarter-wave condition for a single-layer ARC is:

$$n_{\text{ARC}} = \sqrt{n_1 \cdot n_2}, \quad t_{\text{ARC}} = \frac{\lambda_0}{4 \cdot n_{\text{ARC}}}$$

For multi-layer stacks, numerical optimization with COMPASS replaces analytical design rules.

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
            },
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
                "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
            },
            "barl": {
                "layers": [
                    {"thickness": 0.010, "material": "sio2"},
                    {"thickness": 0.025, "material": "hfo2"},
                    {"thickness": 0.015, "material": "sio2"},
                    {"thickness": 0.030, "material": "si3n4"},
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
        "params": {"fourier_order": [9, 9]},
        "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
    },
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01}},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}
```

## Baseline: no BARL vs with BARL

Compare QE with and without the anti-reflective layers:

```python
# With BARL (baseline config)
result_with = SingleRunner.run(base_config)

# Without BARL (remove BARL layers)
config_no_barl = copy.deepcopy(base_config)
config_no_barl["pixel"]["layers"]["barl"]["layers"] = []
result_without = SingleRunner.run(config_no_barl)

from compass.visualization.qe_plot import plot_qe_comparison

fig, ax = plt.subplots(figsize=(10, 6))
plot_qe_comparison(
    results=[result_with, result_without],
    labels=["With BARL", "No BARL"],
    ax=ax,
)
ax.set_title("BARL Effect on QE")
plt.tight_layout()
plt.savefig("barl_vs_no_barl.png", dpi=150)
```

<ThinFilmReflectance />

## Quarter-wave single-layer design

Design a single-layer ARC optimized for 550 nm (green peak):

```python
# Ideal ARC refractive index for SiO2 (n~1.46) to Si (n~4.08) interface
n_sio2 = 1.46
n_si = 4.08
n_ideal = np.sqrt(n_sio2 * n_si)
print(f"Ideal ARC index: {n_ideal:.2f}")  # ~2.44

# Quarter-wave thickness at 550 nm
wl_design = 0.55  # um
t_qw = wl_design / (4 * n_ideal)
print(f"Quarter-wave thickness: {t_qw*1000:.1f} nm")  # ~56 nm

# Si3N4 (n~2.0) is the closest standard material
t_si3n4_qw = wl_design / (4 * 2.0)
print(f"Si3N4 quarter-wave: {t_si3n4_qw*1000:.1f} nm")  # ~69 nm
```

Test this single-layer design:

```python
config_single = copy.deepcopy(base_config)
config_single["pixel"]["layers"]["barl"]["layers"] = [
    {"thickness": 0.069, "material": "si3n4"},
]
result_single = SingleRunner.run(config_single)
```

## Thickness sweep for single-layer BARL

Sweep the Si3N4 thickness to find the optimum:

```python
thicknesses = np.arange(0.020, 0.151, 0.005)  # 20 to 150 nm
avg_green_qe = []

for t in thicknesses:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["barl"]["layers"] = [
        {"thickness": float(t), "material": "si3n4"},
    ]
    result = SingleRunner.run(cfg)
    green_qe = np.mean([
        qe for name, qe in result.qe_per_pixel.items() if name.startswith("G")
    ], axis=0)
    # Average across wavelengths
    avg_green_qe.append(float(np.mean(green_qe)))

plt.figure(figsize=(8, 5))
plt.plot(thicknesses * 1000, avg_green_qe, "o-", linewidth=2)
plt.axvline(69, color="red", linestyle="--", alpha=0.5, label="Quarter-wave (69 nm)")
plt.xlabel("Si3N4 Thickness (nm)")
plt.ylabel("Average Green QE (400-700 nm)")
plt.title("Single-Layer BARL Thickness Sweep")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("barl_thickness_sweep.png", dpi=150)
```

## Multi-layer broadband BARL design

A single layer is optimized for one wavelength. For broadband performance, use alternating high-index and low-index layers:

```python
# Two-layer: HfO2 (high-n) + SiO2 (low-n)
designs = {
    "2-layer HfO2/SiO2": [
        {"thickness": 0.025, "material": "hfo2"},
        {"thickness": 0.035, "material": "sio2"},
    ],
    "3-layer SiO2/HfO2/SiO2": [
        {"thickness": 0.010, "material": "sio2"},
        {"thickness": 0.030, "material": "hfo2"},
        {"thickness": 0.020, "material": "sio2"},
    ],
    "4-layer (baseline)": [
        {"thickness": 0.010, "material": "sio2"},
        {"thickness": 0.025, "material": "hfo2"},
        {"thickness": 0.015, "material": "sio2"},
        {"thickness": 0.030, "material": "si3n4"},
    ],
}

fig, ax = plt.subplots(figsize=(10, 6))

for name, layers in designs.items():
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["barl"]["layers"] = layers
    result = SingleRunner.run(cfg)

    green_qe = np.mean([
        qe for pname, qe in result.qe_per_pixel.items() if pname.startswith("G")
    ], axis=0)

    total_t = sum(l["thickness"] for l in layers) * 1000
    ax.plot(result.wavelengths * 1000, green_qe,
            label=f"{name} ({total_t:.0f} nm total)", linewidth=2)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Green QE")
ax.set_title("Multi-Layer BARL Design Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("barl_multilayer_comparison.png", dpi=150)
```

## HfO2 thickness sweep in a 2-layer stack

Fix SiO2 at 15 nm and sweep HfO2 thickness:

```python
hfo2_thicknesses = np.arange(0.010, 0.061, 0.005)
sio2_thickness = 0.015

broadband_qe = []

for t_hfo2 in hfo2_thicknesses:
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["barl"]["layers"] = [
        {"thickness": float(sio2_thickness), "material": "sio2"},
        {"thickness": float(t_hfo2), "material": "hfo2"},
    ]
    result = SingleRunner.run(cfg)

    # Broadband average QE across all channels
    all_qe = np.mean([qe for qe in result.qe_per_pixel.values()], axis=0)
    broadband_qe.append(float(np.mean(all_qe)))

plt.figure(figsize=(8, 5))
plt.plot(hfo2_thicknesses * 1000, broadband_qe, "s-", linewidth=2, color="tab:orange")
plt.xlabel("HfO2 Thickness (nm)")
plt.ylabel("Broadband Average QE")
plt.title("HfO2 Thickness Sweep (SiO2 = 15 nm)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("barl_hfo2_sweep.png", dpi=150)
```

## Reflectance analysis

Examine the reflectance spectrum to understand BARL effectiveness:

```python
fig, ax = plt.subplots(figsize=(10, 5))

for name, layers in designs.items():
    cfg = copy.deepcopy(base_config)
    cfg["pixel"]["layers"]["barl"]["layers"] = layers
    result = SingleRunner.run(cfg)
    if result.reflection is not None:
        ax.plot(result.wavelengths * 1000, result.reflection,
                label=name, linewidth=2)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_title("BARL Stack Reflectance Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("barl_reflectance.png", dpi=150)
```

<EnergyBalanceDiagram />

## Design guidelines

| Parameter              | Guideline                                           |
|------------------------|------------------------------------------------------|
| Number of layers       | 2--4 layers for broadband visible ARC                |
| Material choices       | SiO2 (low-n), Si3N4 (mid-n), HfO2/TiO2 (high-n)   |
| Total BARL thickness   | 40--120 nm typical                                   |
| Target reflectance     | < 5% across 400--700 nm                              |
| Optimization metric    | Broadband average QE across all channels             |

For production designs, combine BARL optimization with microlens and color filter optimization in a joint parameter sweep.
