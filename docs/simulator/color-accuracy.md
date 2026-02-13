---
title: Color Accuracy Analyzer
---

# Color Accuracy Analyzer

Evaluate color reproduction accuracy of a sensor by computing the Color Correction Matrix and measuring color error against standard ColorChecker patches.

<ColorAccuracyAnalyzer />

## Physics

A sensor's raw RGB response does not directly correspond to human color perception. Color accuracy analysis bridges this gap through a calibrated pipeline:

### Pipeline

1. **Spectral sensitivity** — the sensor's RGB spectral response is convolved with each ColorChecker patch reflectance spectrum under D65 illumination to produce raw RGB triplets.
2. **Color Correction Matrix (CCM)** — a 3x3 matrix maps raw sensor RGB to CIE XYZ tristimulus values. The CCM is computed via least-squares regression against known patch XYZ values.
3. **Delta E (DE\*ab)** — the perceptual color error is computed in CIELAB space. DE\*ab < 1 is imperceptible; DE\*ab > 3 is noticeable.

### What Affects Color Accuracy

- **Filter spectral shape** — narrower filters improve color separation but reduce sensitivity
- **IR contamination** — silicon responds to NIR light that the eye does not see, causing color shifts without an IR-cut filter
- **Illuminant mismatch** — a CCM optimized for D65 may perform poorly under tungsten or fluorescent lighting

::: info
The 24-patch Macbeth ColorChecker is the industry standard target for camera color calibration and the basis for most published sensor color accuracy benchmarks.
:::
