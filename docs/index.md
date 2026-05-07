---
layout: home

hero:
  name: "COMPASS"
  text: "One pixel config. Nine EM solvers. Cross-validated."
  tagline: Open-source CMOS image sensor optics platform — define your pixel stack once and run it through 9 RCWA/FDTD/TMM backends to cross-check QE, crosstalk, and field maps.
  image:
    src: /logo.svg
    alt: COMPASS
  actions:
    - theme: brand
      text: Get Started
      link: /guide/quickstart
    - theme: alt
      text: New to image sensors?
      link: /theory/basics/what-is-cmos-sensor
features:
  - title: "📖 Beginner Friendly"
    details: Start from zero — learn image sensor optics fundamentals before diving into simulation
    link: /theory/basics/what-is-cmos-sensor
---

<HeroAnimation />

## Why COMPASS?

COMPASS bridges the gap between electromagnetic theory and practical CMOS image sensor design. Define your pixel stack once, run it through any solver, and compare results -- all from Python.

<FeatureShowcase />

## Architecture

A clean five-stage pipeline takes you from YAML configuration to publication-ready results. Click any stage to learn more.

<ArchitectureOverview />

## Solver Backends

COMPASS provides a unified interface to **9 solver backends** across three electromagnetic methods. Click any solver to see details.

<SolverShowcase />

## Browser-Based Simulators

In addition to the Python solver pipeline, COMPASS ships with **20+ browser-based simulators** for quick exploration and intuition-building. They run entirely client-side -- no install, no Python required -- and are designed for teaching, design-space exploration, and sanity checks before committing to a full RCWA/FDTD run.

- **Optical stack** -- TMM QE, thin-film designer, energy budget
- **Performance** -- SNR, dynamic range, EMVA 1288, photon transfer curve
- **Wave physics** -- Si absorption, microlens raytrace, Fabry-Pérot, diffraction PSF
- **System** -- MTF, color accuracy (ΔE), pixel scaling, dark current

[Browse all simulators →](/simulator/)

## Quick Example

Define your simulation in a single YAML config and run it with three lines of Python:

```yaml
# config.yaml
pixel:
  pitch: 1.0          # um
  unit_cell: [2, 2]   # 2x2 Bayer pattern

solver:
  name: torcwa
  type: rcwa
  fourier_order: 9

source:
  wavelength:
    mode: sweep
    sweep: { start: 0.4, stop: 0.7, step: 0.01 }
  polarization: unpolarized
```

```python
from compass.runners.single_run import SingleRunner

result = SingleRunner.run("config.yaml")

for pixel, qe in result.qe_per_pixel.items():
    print(f"{pixel}: peak QE = {qe.max():.2%}")
```

<div class="landing-cta-section">

## Get Started

<div class="cta-grid">
<a href="/guide/quickstart" class="cta-card">
  <strong>Quick Start</strong>
  <span>Run your first simulation in minutes</span>
</a>
<a href="/guide/installation" class="cta-card">
  <strong>Installation Guide</strong>
  <span>Set up COMPASS and solver backends</span>
</a>
<a href="/theory/basics/what-is-cmos-sensor" class="cta-card">
  <strong>Image Sensor Basics</strong>
  <span>New to image sensors? Start here</span>
</a>
<a href="/theory/simulation/rcwa-vs-fdtd" class="cta-card">
  <strong>RCWA vs FDTD</strong>
  <span>Pick the right solver for the job</span>
</a>
<a href="/cookbook/bsi-2x2-basic" class="cta-card">
  <strong>Cookbook</strong>
  <span>Practical recipes for common tasks</span>
</a>
</div>

</div>
