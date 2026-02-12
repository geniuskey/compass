---
title: Cone Illumination
description: Setting up cone illumination from a lens exit pupil in COMPASS, including CRA, F-number, angular sampling methods, weighting functions, and integration with planewave solvers.
---

# Cone Illumination

In a real camera system, light reaching each pixel comes from the full area of the lens exit pupil, not from a single direction. The `ConeIllumination` class models this by decomposing the illumination cone into weighted planewaves and integrating the results.

## Physical background

<ConeIlluminationViewer />

The illumination cone is characterized by:

- **CRA (Chief Ray Angle)**: The angle between the optical axis and the central ray from the exit pupil to the pixel. CRA is zero at the sensor center and increases toward the edges.
- **F-number**: Determines the half-cone angle of the illumination via $\theta_{\text{half}} = \arcsin(1 / 2F)$. An F/2.0 lens gives a half-cone of about 14.5 degrees.
- **Weighting**: The intensity distribution across the cone. Light near the cone edge is typically weaker than at the center (e.g., cosine or cos^4 weighting).

The cone illumination result is obtained by running multiple planewave simulations at sampled angles within the cone and then computing the weighted average of the QE.

## Top view: footprint on the pixel array

<ConeIlluminationTopView />

The side view above shows the cone geometry in cross-section. The **top view** provides a complementary perspective: looking down at the pixel array from above, you can see how the illumination cone projects onto the 2x2 Bayer pattern.

Key observations from the top view:

- **Footprint diameter**: The cone footprint on the focal plane has diameter $d = 2 h \tan(\theta_{\text{half}})$, where $h$ is the pixel stack height. A lower F-number produces a wider footprint.
- **CRA shift**: A nonzero CRA shifts the footprint center away from the pixel center. At CRA = 20° on a typical 5 um stack, the shift can exceed 1.5 um — comparable to the pixel pitch itself.
- **Sampling coverage**: The interactive viewer above shows how fibonacci and grid sampling points distribute across the footprint. Fibonacci sampling provides more uniform angular coverage.
- **Lens area**: The footprint area $A = \pi r^2$ where $r = h \tan(\theta_{\text{half}})$ determines how much of the neighboring pixel receives light from the cone, which directly affects crosstalk.

## Creating a ConeIllumination instance

```python
from compass.sources.cone_illumination import ConeIllumination

cone = ConeIllumination(
    cra_deg=15.0,          # Chief Ray Angle in degrees
    f_number=2.0,          # F-number of the lens
    n_points=37,           # Number of angular sample points
    sampling="fibonacci",  # "fibonacci" or "grid"
    weighting="cosine",    # "uniform", "cosine", "cos4", or "gaussian"
)

print(f"Half-cone angle: {cone.half_cone_rad * 180 / 3.14159:.1f} degrees")
```

## Sampling methods

The `get_sampling_points()` method returns a list of `(theta_deg, phi_deg, weight)` tuples. Each tuple represents a planewave direction and its associated integration weight.

### Fibonacci sampling

Fibonacci (golden-angle spiral) sampling distributes points quasi-uniformly over the cone area. It provides good coverage with relatively few points.

```python
cone = ConeIllumination(
    cra_deg=10.0, f_number=2.8, n_points=37, sampling="fibonacci"
)

points = cone.get_sampling_points()
print(f"Number of sample points: {len(points)}")
for i, (theta, phi, w) in enumerate(points[:5]):
    print(f"  Point {i}: theta={theta:.2f} deg, phi={phi:.2f} deg, weight={w:.4f}")
```

Fibonacci sampling is recommended for most cases. Use 19-37 points for quick estimates and 61-91 points for production results.

### Grid sampling

Grid sampling uses a uniform $(\theta, \phi)$ grid. It is straightforward but less efficient than Fibonacci for the same number of points.

```python
cone = ConeIllumination(
    cra_deg=10.0, f_number=2.8, n_points=36, sampling="grid"
)

points = cone.get_sampling_points()
print(f"Grid sampling: {len(points)} points")
```

Grid sampling produces `n_theta x n_phi` points where `n_theta = sqrt(n_points)` and `n_phi = n_points / n_theta`.

## Weighting functions

The weighting function models the intensity distribution across the pupil:

| Weight     | Formula                | Physical model                      |
|------------|------------------------|--------------------------------------|
| `uniform`  | $w = 1$               | Flat pupil illumination              |
| `cosine`   | $w = \cos\theta$       | Lambertian source / aplanatic lens   |
| `cos4`     | $w = \cos^4\theta$     | Cos-fourth falloff at image plane    |
| `gaussian`  | $w = e^{-\theta^2/2\sigma^2}$ | Apodized pupil            |

The default is `cosine`, which is appropriate for most camera lens systems.

```python
# Compare different weighting functions
for wf in ["uniform", "cosine", "cos4", "gaussian"]:
    cone = ConeIllumination(
        cra_deg=0.0, f_number=2.0, n_points=37, weighting=wf
    )
    points = cone.get_sampling_points()
    weights = [p[2] for p in points]
    print(f"{wf:10s}: max_w={max(weights):.4f}, min_w={min(weights):.4f}")
```

## Integrating with planewave solvers

To compute cone-illuminated QE, run a planewave simulation at each sampled angle and compute the weighted sum:

```python
import numpy as np
from compass.sources.cone_illumination import ConeIllumination
from compass.solvers.base import SolverFactory

# Set up cone
cone = ConeIllumination(cra_deg=15.0, f_number=2.0, n_points=37, weighting="cosine")
points = cone.get_sampling_points()

# Create solver
solver = SolverFactory.create("torcwa", solver_config, device="cuda")
solver.setup_geometry(pixel_stack)

# Run planewave at each sample point
wavelength = 0.55
weighted_qe = {}

for theta_deg, phi_deg, weight in points:
    solver.setup_source({
        "wavelength": wavelength,
        "theta": float(theta_deg),
        "phi": float(phi_deg),
        "polarization": "unpolarized",
    })
    result = solver.run()

    for pixel_name, qe in result.qe_per_pixel.items():
        if pixel_name not in weighted_qe:
            weighted_qe[pixel_name] = 0.0
        weighted_qe[pixel_name] += weight * float(np.mean(qe))

print("Cone-illuminated QE at 550 nm:")
for pixel_name, qe in weighted_qe.items():
    print(f"  {pixel_name}: QE = {qe:.3f}")
```

## Cone illumination with wavelength sweep

For a full spectral sweep under cone illumination, iterate over wavelengths and angular samples:

```python
wavelengths = np.arange(0.40, 0.701, 0.01)
cone = ConeIllumination(cra_deg=15.0, f_number=2.0, n_points=37)
points = cone.get_sampling_points()

# Initialize QE storage
pixel_names = None
cone_qe = {}

for wl in wavelengths:
    wl_qe = {}

    for theta_deg, phi_deg, weight in points:
        solver.setup_source({
            "wavelength": float(wl),
            "theta": float(theta_deg),
            "phi": float(phi_deg),
            "polarization": "unpolarized",
        })
        result = solver.run()

        if pixel_names is None:
            pixel_names = list(result.qe_per_pixel.keys())
            for pn in pixel_names:
                cone_qe[pn] = []

        for pn in pixel_names:
            if pn not in wl_qe:
                wl_qe[pn] = 0.0
            wl_qe[pn] += weight * float(np.mean(result.qe_per_pixel[pn]))

    for pn in pixel_names:
        cone_qe[pn].append(wl_qe[pn])

# Plot results
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
wl_nm = wavelengths * 1000
for pn in pixel_names:
    ax.plot(wl_nm, cone_qe[pn], label=pn)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("QE (cone illumination)")
ax.set_title(f"Cone Illumination QE (CRA={cone.cra_deg} deg, F/{cone.f_number})")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

## Sampling convergence

Check that the number of sample points is sufficient by comparing results at increasing `n_points`:

```python
for n in [7, 19, 37, 61, 91]:
    cone = ConeIllumination(cra_deg=15.0, f_number=2.0, n_points=n)
    points = cone.get_sampling_points()
    # ... run weighted sum, record QE
    print(f"n_points={n}: avg QE = ...")
```

Typically, 37 points gives results within 1% of the fully converged integral for F/2.0 and below.

## Lens area sweep: F-number vs QE and crosstalk

The illumination cone footprint area scales with F-number. Sweeping the F-number reveals how lens speed affects QE and optical crosstalk — a critical trade-off in CIS design.

### Footprint area vs F-number

The footprint radius $r = h \tan(\theta_{\text{half}})$ where $\theta_{\text{half}} = \arcsin(1/2F)$. The footprint area $A = \pi r^2$ thus grows rapidly as F-number decreases:

| F-number | $\theta_{\text{half}}$ (deg) | Footprint radius (um) | Area (um²) |
|----------|------|----------|--------|
| F/1.4    | 20.9 | 1.91     | 11.5   |
| F/2.0    | 14.5 | 1.29     | 5.3    |
| F/2.8    | 10.3 | 0.91     | 2.6    |
| F/4.0    | 7.2  | 0.63     | 1.2    |
| F/5.6    | 5.1  | 0.45     | 0.63   |

*(Assuming stack height h = 5.0 um)*

### Running an F-number sweep

```python
import numpy as np
from compass.sources.cone_illumination import ConeIllumination
from compass.solvers.base import SolverFactory

f_numbers = [1.4, 2.0, 2.8, 4.0, 5.6, 8.0]
wavelength = 0.55
cra_deg = 15.0

solver = SolverFactory.create("torcwa", solver_config, device="cuda")
solver.setup_geometry(pixel_stack)

results = {}
for fn in f_numbers:
    cone = ConeIllumination(cra_deg=cra_deg, f_number=fn, n_points=37)
    points = cone.get_sampling_points()

    weighted_qe = {}
    for theta_deg, phi_deg, weight in points:
        solver.setup_source({
            "wavelength": wavelength,
            "theta": float(theta_deg),
            "phi": float(phi_deg),
            "polarization": "unpolarized",
        })
        result = solver.run()

        for pixel_name, qe in result.qe_per_pixel.items():
            if pixel_name not in weighted_qe:
                weighted_qe[pixel_name] = 0.0
            weighted_qe[pixel_name] += weight * float(np.mean(qe))

    results[fn] = weighted_qe
    print(f"F/{fn}: {weighted_qe}")
```

### Analyzing the trade-off

Faster lenses (lower F-number) collect more light, improving signal. But the wider cone also increases angular spread and crosstalk between adjacent pixels:

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# QE vs F-number for the green pixel
green_qe = [results[fn].get("green_tl", 0) for fn in f_numbers]
ax1.plot(f_numbers, green_qe, "go-", linewidth=2, markersize=8)
ax1.set_xlabel("F-number")
ax1.set_ylabel("QE (green pixel)")
ax1.set_title("QE vs F-number (550 nm)")
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

# Crosstalk: ratio of non-target pixel QE to target pixel QE
crosstalk = []
for fn in f_numbers:
    green = results[fn].get("green_tl", 1e-9)
    red = results[fn].get("red_tr", 0)
    xtalk = red / green * 100  # percentage
    crosstalk.append(xtalk)

ax2.plot(f_numbers, crosstalk, "rs-", linewidth=2, markersize=8)
ax2.set_xlabel("F-number")
ax2.set_ylabel("Crosstalk (%)")
ax2.set_title("Green→Red crosstalk vs F-number")
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

plt.tight_layout()
```

### Combined CRA + F-number sweep

For a full lens-area sensitivity study, sweep both CRA and F-number to build a 2D map:

```python
cra_values = [0, 5, 10, 15, 20, 25, 30]
f_numbers = [1.4, 2.0, 2.8, 4.0]
wavelength = 0.55

qe_map = np.zeros((len(cra_values), len(f_numbers)))

for i, cra in enumerate(cra_values):
    for j, fn in enumerate(f_numbers):
        cone = ConeIllumination(cra_deg=cra, f_number=fn, n_points=37)
        points = cone.get_sampling_points()

        weighted_qe = 0.0
        for theta_deg, phi_deg, weight in points:
            solver.setup_source({
                "wavelength": wavelength,
                "theta": float(theta_deg),
                "phi": float(phi_deg),
                "polarization": "unpolarized",
            })
            result = solver.run()
            weighted_qe += weight * float(
                np.mean(result.qe_per_pixel.get("green_tl", [0]))
            )

        qe_map[i, j] = weighted_qe

# Plot as heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(qe_map, aspect="auto", origin="lower",
               extent=[f_numbers[0], f_numbers[-1],
                       cra_values[0], cra_values[-1]])
ax.set_xlabel("F-number")
ax.set_ylabel("CRA (degrees)")
ax.set_title("Green pixel QE: CRA vs F-number")
plt.colorbar(im, ax=ax, label="QE")
plt.tight_layout()
```

This 2D sweep helps identify the CRA and F-number limits for a target QE threshold — essential information for microlens design optimization.

## Configuration via YAML

Cone illumination can be specified in the source config:

```yaml
source:
  type: "cone"
  cone:
    cra_deg: 15.0
    f_number: 2.0
    n_points: 37
    sampling: "fibonacci"
    weighting: "cosine"
  wavelength:
    mode: "sweep"
    sweep: {start: 0.40, stop: 0.70, step: 0.01}
  polarization: "unpolarized"
```

## Next steps

- [ROI Sweep](./roi-sweep.md) -- sweep cone illumination across the sensor
- [Microlens & CRA Optimization cookbook](../cookbook/microlens-optimization.md) -- CRA vs QE study
- [First Simulation](./first-simulation.md) -- basic planewave setup
