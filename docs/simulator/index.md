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

### Optics & Wave Physics
- **[Si Absorption Depth](./si-absorption)** — Visualize Beer-Lambert absorption in silicon and understand wavelength-dependent penetration depth
- **[Microlens Ray Trace](./microlens-raytrace)** — Trace rays through superellipse microlens geometry with Snell's law refraction and CRA effects
- **[Fabry-P&eacute;rot Visualizer](./fabry-perot)** — Explore thin film interference with phasor diagrams and quarter-wave anti-reflection conditions
- **[Diffraction PSF Viewer](./diffraction-psf)** — View Airy patterns, encircled energy, and pixel grid overlay for collection efficiency analysis

### System Analysis
- **[MTF Analyzer](./mtf-analyzer)** — Compute pixel aperture, diffraction, and combined system MTF with Nyquist frequency analysis
- **[Pixel Scaling Trends](./pixel-scaling)** — Explore how FWC, SNR, and diffraction limits scale with pixel pitch
- **[Color Accuracy Analyzer](./color-accuracy)** — Evaluate color reproduction with CCM computation and Delta E analysis on ColorChecker patches
- **[Dark Current & Temperature](./dark-current)** — Simulate Arrhenius dark current model and visualize thermal noise impact on image quality

::: tip
These simulators use a 1D TMM approximation. For full 3D simulations with RCWA or FDTD, see the [Guide](/guide/installation).
:::
