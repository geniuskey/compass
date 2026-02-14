---
title: MLA Array Visualizer
---

# MLA Array Visualizer

Visualize micro lens array (MLA) geometry with configurable array patterns, asymmetric lens radii, and curvature parameters. Switch between contour maps, equal-aspect cross-sections, 3D wireframe surface, and 2D ray tracing views.

<MlaArrayVisualizer />

## Physics

Each microlens is modeled as a superellipse profile with independent X and Y radii, allowing asymmetric lens shapes. The height profile follows:

**z(r) = h (1 - r^2)^{1/(2alpha)}**

where the normalized radial distance r is computed from the superellipse equation:

**r = (|x/Rx|^n + |y/Ry|^n)^{1/n}**

### Key Parameters

- **Rx, Ry** — Semi-axes of the superellipse footprint. When Rx = Ry the lens is circularly symmetric; different values create an anamorphic lens
- **Shape parameter n** — Controls the boundary shape: n=2 gives an ellipse, n>2 produces a more rectangular footprint (supercircle)
- **Curvature alpha** — Adjusts the sag profile steepness. alpha=1 gives a standard parabolic-like shape; alpha<1 creates a flatter top, alpha>1 creates a sharper peak
- **Spacing** — Pitch between lens centers in the array. Fill factor depends on the ratio of lens area to cell area

### Views

- **Contour** — Top-down iso-height contour lines showing the lens footprint shape across the array
- **Cross-Section** — XZ, YZ, and diagonal profiles with equal x/y scaling, useful for comparing asymmetric shapes
- **3D Surface** — Interactive wireframe rendering with adjustable elevation and azimuth angles
- **Ray Trace** — 2D geometric ray tracing through the XZ plane with Snell's law refraction at the lens surface

::: info
This simulator focuses on MLA geometry and array patterns. For single-pixel ray tracing with full pixel stack (silicon, BARL, color filter, planarization), see [Microlens Ray Trace](./microlens-raytrace).
:::
