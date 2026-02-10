---
layout: home

hero:
  name: "COMPASS"
  text: "Cross-solver Optical Modeling Platform for Advanced Sensor Simulation"
  tagline: Simulate quantum efficiency of BSI CMOS image sensor pixels with RCWA and FDTD solvers
  actions:
    - theme: brand
      text: Get Started
      link: /guide/installation
    - theme: alt
      text: Theory Background
      link: /theory/light-basics
    - theme: alt
      text: View on GitHub
      link: https://github.com/compass-sim/compass

features:
  - title: Multi-Solver Support
    details: Run the same pixel structure through RCWA (torcwa, grcwa, meent) and FDTD (fdtd, meep) solvers and compare results head-to-head.
  - title: Parametric Pixel Modeling
    details: Define complete BSI pixel stacks -- microlens, color filters, BARL, silicon, DTI -- in a single YAML config and sweep any parameter.
  - title: Built-in Visualization
    details: Plot QE spectra, field distributions, cross-talk matrices, and 3D structure views out of the box with matplotlib and PyVista.
  - title: Numerical Stability
    details: Five-layer stability defense including mixed-precision eigendecomposition, S-matrix recursion, Li factorization, and adaptive fallback.
---

## What is COMPASS?

COMPASS is a Python framework for simulating the optical performance of backside-illuminated (BSI) CMOS image sensor pixels. It bridges the gap between electromagnetic (EM) theory and practical sensor design by providing a unified interface to multiple solver backends.

Given a pixel stack definition (microlens geometry, color filter pattern, anti-reflection coatings, silicon photodiode), COMPASS computes the **quantum efficiency (QE)** -- the fraction of incident photons that generate electron-hole pairs in each photodiode -- across wavelength, angle, and polarization.

### Typical workflow

```
YAML config  -->  PixelStack  -->  Solver (RCWA / FDTD)  -->  QE spectrum
                                                           -->  Field maps
                                                           -->  Energy balance
```

## Quick example

```python
from compass.runners.single_run import SingleRunner

result = SingleRunner.run({
    "pixel": {"pitch": 1.0, "unit_cell": [2, 2]},
    "solver": {"name": "torcwa", "type": "rcwa"},
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.4, "stop": 0.7, "step": 0.01}},
        "polarization": "unpolarized",
    },
})

# result.qe_per_pixel is a dict mapping pixel names to QE arrays
for pixel, qe in result.qe_per_pixel.items():
    print(f"{pixel}: peak QE = {qe.max():.2%}")
```
