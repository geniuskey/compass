# Theory

This section covers the physics, sensor architecture, and numerical methods behind every COMPASS simulation. The chapters are organized so you can read top-to-bottom or jump to the topic you need.

## How this section is organized

| Chapter | Scope | Read this if you want to... |
|---|---|---|
| [Optics](./optics-intro.md) | Wave optics fundamentals — fields, films, diffraction | Understand *why* a wave-optics solver is needed for sub-wavelength pixels |
| [Image Sensor](./sensor-intro.md) | BSI pixel structure, QE, full radiometric signal chain | Understand *what* is being modeled and how the result is interpreted |
| [Simulation](./simulation-intro.md) | RCWA, FDTD, TMM and their numerical behavior | Understand *how* the solver actually computes the answer |

## Suggested reading paths

**Newcomer to image sensor optics**
1. [Light Basics](./light-basics.md) → [Electromagnetic Waves](./electromagnetic-waves.md) → [Thin Film Optics](./thin-film-optics.md) → [Diffraction](./diffraction.md)
2. [Image Sensor Optics](./image-sensor-optics.md) → [Quantum Efficiency](./quantum-efficiency.md)
3. [Simulation Intro](./simulation-intro.md) → pick one solver page

**Engineer evaluating COMPASS for a project**
1. [Image Sensor Optics](./image-sensor-optics.md) → [Quantum Efficiency](./quantum-efficiency.md) → [Signal Chain](./signal-chain.md)
2. [Simulation Intro](./simulation-intro.md) → [RCWA vs FDTD](./rcwa-vs-fdtd.md) → [Numerical Stability](./numerical-stability.md)

**Researcher implementing a new solver or analysis**
1. [Optics Intro](./optics-intro.md) → [Electromagnetic Waves](./electromagnetic-waves.md)
2. [RCWA Explained](./rcwa-explained.md) and [FDTD Explained](./fdtd-explained.md)
3. [Numerical Stability](./numerical-stability.md)

::: tip
If you have not used COMPASS before, start with [Introduction → What is a CMOS Image Sensor?](/introduction/what-is-cmos-sensor) for a non-mathematical orientation, then come back here.
:::
