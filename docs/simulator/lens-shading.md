---
title: "Lens Shading Simulator"
---

# Lens Shading Simulator

Simulate relative illumination across the sensor based on chief ray angle, microlens shift optimization, and cos-4 fall-off. Visualize per-channel color shading.

<LensShadingSimulator />

## Relative Illumination Model

The relative illumination (RI) at each sensor position combines geometric and optical effects:

**RI(r) = cos^4(CRA) x ML_coupling x CF_shift**

### cos^4 Law
Geometric vignetting follows the fourth power of cosine of the chief ray angle, causing natural brightness fall-off from center to edge.

### Microlens Coupling
When the chief ray angle exceeds the microlens acceptance cone, light collection efficiency drops. The ML shift ratio parameter controls how well the microlens array compensates for the CRA by shifting microlens positions toward the optical axis.

### Color Shading
Different color channels experience different shading because:
- Blue light has shorter absorption depth, making it more sensitive to angular coupling
- Red light penetrates deeper into silicon, reducing angular sensitivity
- This creates a color cast from center to edge that must be corrected in the ISP

::: tip
A ML shift ratio of 1.0 means perfect CRA compensation. Real sensors typically achieve 0.6-0.8 due to manufacturing constraints.
:::
