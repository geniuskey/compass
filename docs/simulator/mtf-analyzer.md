---
title: MTF Analyzer
---

# MTF Analyzer

Analyze the Modulation Transfer Function of an image sensor system, combining pixel aperture, optical diffraction, and defocus contributions to understand spatial resolution limits.

<MtfAnalyzer />

## Physics

The MTF describes how faithfully a system reproduces spatial detail at each frequency. For a sensor, the total MTF is the product of individual contributions:

**MTF_total(f) = MTF_pixel(f) x MTF_optics(f)**

### Components

- **Pixel aperture MTF** = |sinc(f x pitch)| — the finite pixel size acts as a spatial averaging filter. Smaller pixels preserve higher frequencies.
- **Optical diffraction MTF** — for a circular aperture, the MTF rolls off as a function of f/(lambda x f-number). The cutoff frequency is 1/(lambda x F/#).
- **Combined system MTF** — the product of all components determines the effective resolution.

### Nyquist and Aliasing

The pixel sampling frequency is **f_Nyquist = 1/(2 x pitch)**. Spatial frequencies above Nyquist are aliased, producing moire artifacts. The simulator shows how pitch, wavelength, and f-number interact to determine whether the system is diffraction-limited or pixel-limited.

::: tip
A well-designed system balances optical MTF cutoff near the pixel Nyquist frequency — oversampling wastes pixels, while undersampling creates aliasing.
:::
