---
title: Theory Coverage Map
description: Coverage audit for COMPASS theory documentation, showing where each CMOS image sensor optics topic is explained and which extension points remain.
---

# Theory Coverage Map

This page is a coverage audit for the public theory documentation. It is meant to help maintainers and advanced readers see where each topic belongs without turning individual theory pages into planning documents.

## Current Coverage

| Area | Covered in | Status |
|---|---|---|
| CMOS image sensor orientation | [Basics](../theory/basics/) | Stable entry point |
| Plain-language optics | [Optics Primer](../theory/basics/optics-primer.md) | Stable entry point |
| Wave-optics formulas | [Optics](../theory/optics/) | Core coverage present |
| Pixel stack anatomy | [Image Sensor Optics](../theory/sensor/image-sensor-optics.md) | Core coverage present |
| Pixel-level optical trade-offs | [Pixel Optical Effects](../theory/sensor/pixel-optical-effects.md) | Core coverage present |
| QE and crosstalk metrics | [Quantum Efficiency](../theory/sensor/quantum-efficiency.md) | Core coverage present |
| Radiometric signal chain | [Signal Chain](../theory/sensor/signal-chain.md) | Core coverage present |
| Solver fundamentals | [Optical Simulation](../theory/simulation/) | Core coverage present |
| RCWA/FDTD comparison | [RCWA vs FDTD](../theory/simulation/rcwa-vs-fdtd.md) | Core coverage present |
| Validation evidence | [Reports](../reports/) | Generated evidence published separately |

## Topic Ownership

| Topic | Owner page | Boundary |
|---|---|---|
| Microlens shape and CRA | [Pixel Optical Effects](../theory/sensor/pixel-optical-effects.md) | Cookbook pages show runnable sweeps |
| CFA spectral behavior | [Pixel Optical Effects](../theory/sensor/pixel-optical-effects.md) | Signal Chain handles illuminants and color metrics |
| BARL and ARC | [Thin Film Optics](../theory/optics/thin-film-optics.md), [Pixel Optical Effects](../theory/sensor/pixel-optical-effects.md) | Optics explains films; Sensor explains design impact |
| DTI and crosstalk | [Pixel Optical Effects](../theory/sensor/pixel-optical-effects.md), [Quantum Efficiency](../theory/sensor/quantum-efficiency.md) | QE page defines the matrix |
| Angular and polarization response | [Pixel Optical Effects](../theory/sensor/pixel-optical-effects.md), [Light Basics](../theory/optics/light-basics.md) | Simulation pages explain solver handling |
| Solver convergence | [Numerical Stability](../theory/simulation/numerical-stability.md), [Reports](../reports/convergence-analysis.md) | Reports contain generated numbers |

## Extension Points

These are the next high-value content areas if the theory section needs more depth:

| Extension | Best location | Reason |
|---|---|---|
| Electrical collection and carrier diffusion | New sensor page | Optical QE is an upper bound without charge-collection modeling |
| Process variation and tolerance analysis | Guide or research note | Best presented as workflow plus statistical interpretation |
| Lens shading and module-level CRA maps | Sensor or simulator page | Bridges pixel optics and camera module assumptions |
| MTF and optical crosstalk relationship | Sensor or signal-chain page | Connects pixel leakage to image-level sharpness |
| Calibration against measured QE | Reports or validation guide | Should be tied to data provenance and measurement conditions |

## Maintenance Rule

When adding content, keep the page roles separate: Basics builds intuition, Optics defines physical laws, Sensor maps those laws onto pixel components, Simulation explains numerical methods, Guide gives commands, and Reports publish generated evidence.
