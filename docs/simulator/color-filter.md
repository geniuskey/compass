---
title: Color Filter Designer
---

# Color Filter Designer

Design RGB color filter spectral responses and visualize the resulting color gamut on a CIE 1931 chromaticity diagram. Adjust center wavelengths, bandwidths, and peak transmissions to optimize color reproduction.

<ColorFilterDesigner />

## Color Filter Design Trade-offs

- **Narrower bandwidth** → purer colors, larger gamut, but lower QE (less light passes through)
- **Wider bandwidth** → higher QE, but more spectral overlap between channels (crosstalk)
- **Center wavelength** placement determines the gamut shape and coverage relative to sRGB

::: info
The sRGB gamut triangle is shown for reference. Real image sensors typically achieve 70-90% of sRGB gamut area depending on the color filter design.
:::
