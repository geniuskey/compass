---
layout: home
title: COMPASS — Open-source CMOS image sensor optics platform
titleTemplate: false

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
    - theme: alt
      text: GitHub
      link: https://github.com/geniuskey/compass

features:
  - icon: 📖
    title: Learn the physics
    details: Image sensor optics from first principles — pixel anatomy, thin films, diffraction, QE, RCWA and FDTD.
    link: /theory/
    linkText: Start learning
  - icon: 🐍
    title: Run the Python pipeline
    details: One YAML config, nine solver backends. Get per-pixel QE, crosstalk and field maps you can compare.
    link: /guide/
    linkText: Read the guide
  - icon: 🧪
    title: Try it in the browser
    details: 26 client-side simulators for TMM QE, SNR, MTF, microlens ray tracing and more — no install required.
    link: /simulator/
    linkText: Open simulators
  - icon: ✅
    title: Check the evidence
    details: Cross-solver parity, convergence studies and benchmarks, with the scripts that produced them.
    link: /reports/
    linkText: See reports
---

<div class="home-stats">
  <div class="home-stat"><strong>9</strong><span>EM solver backends</span></div>
  <div class="home-stat"><strong>3</strong><span>methods: RCWA · FDTD · TMM</span></div>
  <div class="home-stat"><strong>26</strong><span>browser simulators</span></div>
  <div class="home-stat"><strong>EN · KO</strong><span>bilingual documentation</span></div>
</div>

<div class="home-eyebrow">The problem</div>

## From photon to electron

Light passes through a microlens, color filter and anti-reflection stack before it is absorbed in silicon. At sub-micron pitch every layer is a wave-optics problem — and different solvers can disagree. COMPASS lets you model the stack once and check the answer against several independent methods.

<HeroAnimation />

<div class="home-eyebrow">Why COMPASS</div>

## Built for cross-validated pixel optics

<FeatureShowcase />

<div class="home-eyebrow">Architecture</div>

## A five-stage pipeline

From YAML configuration to publication-ready results. Click any stage to learn more.

<ArchitectureOverview />

<div class="home-eyebrow">Backends</div>

## Nine solvers, one interface

Three electromagnetic methods behind a single `SolverBase` API. Click any solver to see details.

<SolverShowcase />

<div class="home-eyebrow">Quick example</div>

## Define once, run anywhere

Describe the pixel in YAML, then run it with a few lines of Python. Switch `solver.name` to cross-check the same structure with another backend.

::: code-group

```yaml [config.yaml]
pixel:
  pitch: 1.0          # um
  unit_cell: [2, 2]   # 2x2 Bayer pattern

solver:
  name: torcwa        # or grcwa, meent, fmmax, fdtd_flaport, meep, tmm ...
  type: rcwa
  params:
    fourier_order: [9, 9]

source:
  wavelength:
    mode: sweep
    sweep: { start: 0.40, stop: 0.70, step: 0.01 }
  polarization: unpolarized
```

```python [run.py]
import yaml
from compass.runners.single_run import SingleRunner

with open("config.yaml") as f:
    config = yaml.safe_load(f)

result = SingleRunner.run(config)

for pixel, qe in result.qe_per_pixel.items():
    print(f"{pixel}: peak QE = {qe.max():.2%}")
```

```bash [CLI (Hydra)]
python scripts/run_simulation.py solver=torcwa source=wavelength_sweep
python scripts/run_simulation.py solver=grcwa  source=wavelength_sweep  # cross-check
```

:::

<div class="landing-cta-section">

## Get started

Pick an entry point — every page has a Korean translation via the language menu.

<HomeCta />

</div>
