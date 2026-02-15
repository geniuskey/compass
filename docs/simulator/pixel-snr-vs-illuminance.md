---
title: "Pixel SNR vs Illuminance"
---

# Pixel SNR vs Illuminance

Plot signal-to-noise ratio as a function of photon count (illuminance). Visualize noise breakdown by source and compare actual sensor performance against the ideal shot-noise limit.

<PixelSNRvsIlluminance />

## SNR Model

**SNR = N / sqrt(σ_read² + N + (PRNU × N)²)**

Where:
- **N** — signal electrons (photon count × QE)
- **σ_read** — read noise (electrons RMS)
- **PRNU** — photo response non-uniformity factor

### Noise Regions

| Region | Dominant Noise | SNR Slope |
|--------|---------------|-----------|
| Low light | Read noise | SNR ∝ N |
| Mid range | Shot noise | SNR ∝ √N |
| High signal | PRNU | SNR saturates |

### Key Markers
- **Unity SNR (0 dB)** — signal equals noise, minimum useful exposure
- **20 dB threshold** — often considered minimum for acceptable image quality
- **Saturation** — full well capacity reached

::: tip
The gap between actual and ideal (shot-noise-limited) SNR curves reveals how much the sensor electronics degrade performance. A narrower gap indicates better sensor design.
:::
