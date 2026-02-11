---
layout: home

hero:
  name: "COMPASS"
  text: "Cross-solver Optical Modeling Platform for Advanced Sensor Simulation"
  tagline: Simulate CMOS image sensor pixels with multiple EM solvers from a single YAML config
  image:
    src: /logo.svg
    alt: COMPASS
  actions:
    - theme: brand
      text: Start Here — Image Sensor Basics
      link: /introduction/what-is-cmos-sensor
    - theme: alt
      text: Get Started
      link: /guide/installation
    - theme: alt
      text: View on GitHub
      link: https://github.com/geniuskey/compass
features:
  - title: "\U0001F4D6 Beginner Friendly"
    details: Start from zero — learn image sensor optics fundamentals before diving into simulation
    link: /introduction/what-is-cmos-sensor
---

<HeroAnimation />

## Why COMPASS?

COMPASS bridges the gap between electromagnetic theory and practical CMOS image sensor design. Define your pixel stack once, run it through any solver, and compare results -- all from Python.

<FeatureShowcase />

## Architecture

A clean five-stage pipeline takes you from YAML configuration to publication-ready results. Click any stage to learn more.

<ArchitectureOverview />

## Solver Backends

COMPASS provides a unified interface to **8 solver backends** across three electromagnetic methods. Click any solver to see details.

<SolverShowcase />

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
<a href="/introduction/what-is-cmos-sensor" class="cta-card">
  <strong>Image Sensor Basics</strong>
  <span>New to image sensors? Start here</span>
</a>
<a href="/guide/installation" class="cta-card">
  <strong>Installation Guide</strong>
  <span>Set up COMPASS and solver backends</span>
</a>
<a href="/guide/quickstart" class="cta-card">
  <strong>Quick Start</strong>
  <span>Run your first simulation in minutes</span>
</a>
<a href="/theory/light-basics" class="cta-card">
  <strong>Theory Background</strong>
  <span>Understand the physics behind the simulation</span>
</a>
<a href="/cookbook/bsi-2x2-basic" class="cta-card">
  <strong>Cookbook</strong>
  <span>Practical recipes for common tasks</span>
</a>
</div>

</div>
