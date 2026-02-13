---
title: Simulator
---

# Interactive Simulators

Explore CMOS image sensor pixel optics through browser-based simulators. All calculations run locally using the Transfer Matrix Method (TMM) — no server required.

## Available Simulators

### Optical Stack Analysis
- **[TMM QE Calculator](./tmm-qe)** — Configure a BSI pixel layer stack and compute quantum efficiency spectra in real time
- **[BARL Optimizer](./barl-optimizer)** — Tune anti-reflection coating layer thicknesses to minimize reflectance
- **[Energy Budget Analyzer](./energy-budget)** — Visualize where photon energy goes at each wavelength

### Performance Analysis
- **[Angular Response](./angular-response)** — Study how QE changes with angle of incidence (CRA effects)
- **[SNR Calculator](./snr-calculator)** — Compute signal-to-noise ratio, dynamic range, and photon transfer curves

### Design Tools
- **[Color Filter Designer](./color-filter)** — Design RGB filter spectra and visualize CIE chromaticity gamut
- **[Pixel Design Playground](./pixel-playground)** — Comprehensive pixel designer combining all parameters with multi-panel results

::: tip
These simulators use a 1D TMM approximation. For full 3D simulations with RCWA or FDTD, see the [Guide](/guide/installation).
:::
