---
title: Microlens Ray Trace
---

# Microlens Ray Trace

Trace rays through a superellipse microlens to visualize focusing behavior, spot size, and optical crosstalk as a function of lens geometry and chief ray angle.

<MicrolensRayTrace />

## Physics

The microlens is modeled as a superellipse profile, where the shape parameter controls the transition from a spherical lens to a more cylindrical form. Ray propagation follows Snell's law at each interface:

**n₁ sin(θ₁) = n₂ sin(θ₂)**

### Key Parameters

- **Lens height** determines the focal length — taller lenses focus more tightly but are harder to fabricate
- **Radius of curvature** controls the numerical aperture and collection efficiency
- **Shape parameter** (superellipse exponent) adjusts the lens profile between circular and rectangular
- **CRA angle** simulates off-axis illumination from the camera lens; higher CRA shifts the focal spot and increases crosstalk into neighboring pixels

### Crosstalk Mechanisms

Optical crosstalk occurs when focused light spills into adjacent pixels. The simulator shows how CRA-induced focal shift and lens aberrations contribute to this effect, motivating the use of CRA-dependent microlens offset in real sensor designs.

::: info
This is a geometric (ray) optics approximation. For pixels near or below the diffraction limit, wave-optics effects become significant — use RCWA or FDTD solvers for those cases.
:::
