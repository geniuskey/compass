---
title: CMOS Image Sensor Optics Theory
description: Learning path for CMOS image sensor optics, wave optics, quantum efficiency, signal chains, RCWA, FDTD, and numerical stability.
---

# Theory

This section covers the physics, sensor architecture, and numerical methods behind every COMPASS simulation. The chapters are organized so you can read top-to-bottom or jump to the topic you need.

Use this section for concepts and mental models. Use [Guide](/guide/) for runnable workflows, [Reports](/reports/) for generated validation evidence, and [API Reference](/reference/) for exact classes and configuration fields.

## How this section is organized

| Chapter | Scope | Read this if you want to... |
|---|---|---|
| [Basics](./basics/what-is-cmos-sensor.md) | Non-mathematical orientation to CIS pixels, optics, and QE | Build intuition before the deeper theory pages |
| [Optics](./optics/) | Wave optics fundamentals: fields, films, diffraction | Understand *why* a wave-optics solver is needed for sub-wavelength pixels |
| [Image Sensor](./sensor/) | BSI pixel structure, QE, full radiometric signal chain | Understand *what* is being modeled and how the result is interpreted |
| [Simulation](./simulation/) | RCWA, FDTD, TMM and their numerical behavior | Understand *how* the solver actually computes the answer |

## What This Section Does Not Cover

- Installation and command-line workflows live in [Guide](/guide/).
- Generated benchmark evidence lives in [Reports](/reports/).
- API signatures and YAML schema details live in [API Reference](/reference/).

## Suggested reading paths

**Newcomer to image sensor optics**
1. [What is a CMOS Image Sensor?](./basics/what-is-cmos-sensor.md) -> [Optics Primer](./basics/optics-primer.md) -> [Pixel Anatomy](./basics/pixel-anatomy.md)
2. [Light Basics](./optics/light-basics.md) -> [Electromagnetic Waves](./optics/electromagnetic-waves.md) -> [Thin Film Optics](./optics/thin-film-optics.md) -> [Diffraction](./optics/diffraction.md)
3. [Image Sensor Optics](./sensor/image-sensor-optics.md) -> [Quantum Efficiency](./sensor/quantum-efficiency.md)
4. [Simulation Intro](./simulation/) -> pick one solver page

**Engineer evaluating COMPASS for a project**
1. [Image Sensor Optics](./sensor/image-sensor-optics.md) -> [Quantum Efficiency](./sensor/quantum-efficiency.md) -> [Signal Chain](./sensor/signal-chain.md)
2. [Simulation Intro](./simulation/) -> [RCWA vs FDTD](./simulation/rcwa-vs-fdtd.md) -> [Numerical Stability](./simulation/numerical-stability.md)

**Researcher implementing a new solver or analysis**
1. [Optics Intro](./optics/) -> [Electromagnetic Waves](./optics/electromagnetic-waves.md)
2. [RCWA Explained](./simulation/rcwa-explained.md) and [FDTD Explained](./simulation/fdtd-explained.md)
3. [Numerical Stability](./simulation/numerical-stability.md)

::: tip
If you have not used COMPASS before, start with [Basics -> What is a CMOS Image Sensor?](/theory/basics/what-is-cmos-sensor) for a non-mathematical orientation, then come back here.
:::
