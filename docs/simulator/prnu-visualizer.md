---
title: "PRNU / DSNU Visualizer"
---

# PRNU / DSNU Visualizer

Visualize fixed pattern noise (FPN) in image sensors — both photo response non-uniformity (PRNU) and dark signal non-uniformity (DSNU). Observe spatial noise patterns and statistical distributions.

<PRNUVisualizer />

## Fixed Pattern Noise

Fixed pattern noise (FPN) is a pixel-to-pixel variation that remains constant across frames. It consists of two components:

### PRNU (Photo Response Non-Uniformity)
Gain variation between pixels, proportional to signal level:

**σ_PRNU = PRNU_factor × Signal**

Caused by variations in pixel geometry, microlens alignment, and photodiode doping. Typically 0.5–2% in modern sensors.

### DSNU (Dark Signal Non-Uniformity)
Offset variation between pixels, independent of signal:

**σ_DSNU = DSNU [DN RMS]**

Caused by dark current variation from crystal defects and interface traps.

### Total FPN

**σ_FPN = sqrt(σ_PRNU² + σ_DSNU²)**

::: tip
PRNU dominates at high signal levels, while DSNU dominates in dark regions. FPN can be corrected by per-pixel calibration (flat-field and dark-frame subtraction).
:::
