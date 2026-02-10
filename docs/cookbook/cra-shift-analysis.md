# CRA Shift Analysis

This recipe sweeps Chief Ray Angle from 0 to 30 degrees, compares QE with and without microlens shift compensation, and quantifies the shift's impact on pixel performance.

## Background

At the sensor edge, the chief ray arrives at an oblique angle. Without microlens shift, the focused spot misses the photodiode, causing QE loss and crosstalk. Modern image sensors offset the microlens toward the optical axis to compensate. This recipe quantifies the effect.

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

## CRA sweep: no shift vs auto shift

Run each CRA angle twice -- once without microlens shift and once with automatic CRA-based shift:

```python
cra_angles = np.arange(0, 31, 5)  # 0, 5, 10, ..., 30 degrees
qe_no_shift = []
qe_with_shift = []

for cra in cra_angles:
    # Without microlens shift
    cfg = copy.deepcopy(base_config)
    cfg["source"]["angle"] = {"theta_deg": float(cra), "phi_deg": 0.0}
    cfg["pixel"]["layers"]["microlens"]["shift"]["mode"] = "none"
    result = SingleRunner.run(cfg)
    avg_qe = np.mean([float(np.mean(qe)) for qe in result.qe_per_pixel.values()])
    qe_no_shift.append(avg_qe)

    # With auto CRA shift
    cfg2 = copy.deepcopy(cfg)
    cfg2["pixel"]["layers"]["microlens"]["shift"]["mode"] = "auto_cra"
    cfg2["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = float(cra)
    result2 = SingleRunner.run(cfg2)
    avg_qe2 = np.mean([float(np.mean(qe)) for qe in result2.qe_per_pixel.values()])
    qe_with_shift.append(avg_qe2)

    print(f"CRA={cra:2d} deg: no shift QE={avg_qe:.3f}, with shift QE={avg_qe2:.3f}")
```

## Plot CRA response

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

ax1.plot(cra_angles, qe_no_shift, "o-", label="No ML shift", linewidth=2, color="tab:red")
ax1.plot(cra_angles, qe_with_shift, "s-", label="Auto CRA shift", linewidth=2, color="tab:blue")
ax1.set_ylabel("Average QE at 550 nm")
ax1.set_title("CRA Response: Microlens Shift Compensation")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# QE improvement from shift
improvement = np.array(qe_with_shift) - np.array(qe_no_shift)
ax2.bar(cra_angles, improvement, width=3, color="tab:green", alpha=0.7)
ax2.set_xlabel("Chief Ray Angle (degrees)")
ax2.set_ylabel("QE improvement")
ax2.set_title("QE Gain from Microlens Shift")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cra_shift_analysis.png", dpi=150)
```

## Per-channel CRA response

Examine how each color channel is affected differently:

```python
channels = {"R": "red", "G": "green", "B": "blue"}
channel_qe = {ch: {"no_shift": [], "with_shift": []} for ch in channels}

for cra in cra_angles:
    for mode, shift_mode in [("no_shift", "none"), ("with_shift", "auto_cra")]:
        cfg = copy.deepcopy(base_config)
        cfg["source"]["angle"] = {"theta_deg": float(cra), "phi_deg": 0.0}
        cfg["pixel"]["layers"]["microlens"]["shift"]["mode"] = shift_mode
        if shift_mode == "auto_cra":
            cfg["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = float(cra)
        result = SingleRunner.run(cfg)

        for ch in channels:
            ch_qe = [float(np.mean(qe)) for name, qe in result.qe_per_pixel.items()
                      if name.startswith(ch)]
            channel_qe[ch][mode].append(np.mean(ch_qe) if ch_qe else 0.0)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (ch, color) in zip(axes, channels.items()):
    ax.plot(cra_angles, channel_qe[ch]["no_shift"], "o--",
            label="No shift", color=color, alpha=0.6)
    ax.plot(cra_angles, channel_qe[ch]["with_shift"], "s-",
            label="With shift", color=color)
    ax.set_xlabel("CRA (degrees)")
    ax.set_ylabel("QE at 550 nm")
    ax.set_title(f"{ch} Channel")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

plt.suptitle("Per-Channel CRA Response", fontsize=14)
plt.tight_layout()
plt.savefig("cra_per_channel.png", dpi=150)
```

## Spectral CRA analysis

For a complete picture, sweep both wavelength and CRA:

```python
wavelengths = np.arange(0.40, 0.701, 0.02)  # 20 nm step for speed
cra_list = [0, 10, 20, 30]

fig, ax = plt.subplots(figsize=(10, 6))

for cra in cra_list:
    cfg = copy.deepcopy(base_config)
    cfg["source"] = {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.02}},
        "angle": {"theta_deg": float(cra), "phi_deg": 0.0},
        "polarization": "unpolarized",
    }
    cfg["pixel"]["layers"]["microlens"]["shift"]["mode"] = "auto_cra"
    cfg["pixel"]["layers"]["microlens"]["shift"]["cra_deg"] = float(cra)

    result = SingleRunner.run(cfg)

    green_qe = np.mean([
        qe for name, qe in result.qe_per_pixel.items() if name.startswith("G")
    ], axis=0)
    ax.plot(result.wavelengths * 1000, green_qe, label=f"CRA={cra} deg", linewidth=2)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Green QE")
ax.set_title("Green QE Spectrum at Various CRA (with ML shift)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cra_spectral.png", dpi=150)
```

## Expected results

- **0--10 deg CRA**: QE remains within 5% of normal incidence when ML shift is applied.
- **10--20 deg CRA**: QE begins to drop. ML shift recovers 5--10% absolute QE compared to unshifted.
- **20--30 deg CRA**: Significant QE loss even with shift. Short wavelengths (blue) are affected most because the small absorption depth makes them sensitive to focus offset.
- **Blue channel** is most sensitive to CRA because it absorbs near the surface where microlens focusing accuracy matters most.
- **Red channel** is least sensitive because photons penetrate deep into silicon regardless of focus quality.

## Practical note on shift tables

Real sensor designs use a polynomial or lookup table for the microlens shift versus image height, calibrated during lens-sensor co-design. The `auto_cra` mode provides a first-order approximation. For production designs, you can supply a custom shift table through the `shift.table` config option.
