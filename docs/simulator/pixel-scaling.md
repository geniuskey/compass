---
title: Pixel Scaling Trends
---

# Pixel Scaling Trends

Explore how key image sensor performance metrics scale as pixel pitch shrinks, with reference data from commercial sensors.

<PixelScalingTrends />

## Physics

Pixel miniaturization involves fundamental trade-offs governed by geometry and diffraction:

### Scaling Laws

- **Full Well Capacity:** FWC is proportional to pitch squared (area), so halving the pitch reduces FWC by ~4x, directly impacting dynamic range.
- **Signal-to-Noise Ratio:** For photon-shot-noise-limited conditions, SNR scales with pitch (proportional to the square root of the signal).
- **Nyquist frequency:** f_Nyquist = 1/(2 x pitch) increases with smaller pixels, improving spatial resolution up to the diffraction limit.
- **Diffraction-limited QE:** As the pixel aperture approaches the Airy disk diameter (~2.44 x lambda x F/#), diffraction causes light to spill into neighboring pixels, reducing effective QE.

### Practical Implications

Modern smartphone sensors use pixels from 0.56 um to 1.0 um. At these sizes, microlens optimization, deep-trench isolation, and computational photography become essential to maintain acceptable image quality despite the fundamental scaling penalties.

::: info
Reference data points are approximate values from published sensor specifications and serve as a guide for understanding trends across technology generations.
:::
