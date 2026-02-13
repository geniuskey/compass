---
title: SNR Calculator
---

# SNR Calculator

Compute the signal-to-noise ratio for a CMOS image sensor pixel under various operating conditions. Understand the interplay between quantum efficiency, pixel size, noise sources, and scene illumination.

<SnrCalculator />

## Noise Sources

| Source | Formula | Dominant Regime |
|--------|---------|-----------------|
| Shot noise | √(signal) | Bright scenes |
| Dark current | √(I_dark × t_int) | Long exposures |
| Read noise | constant | Low light |

**Total noise** = √(shot² + dark² + read²)

### Key Metrics
- **SNR (dB)** = 20 × log₁₀(signal / total_noise)
- **Dynamic range** = 20 × log₁₀(full_well / read_noise) — the ratio between the brightest and dimmest signals a pixel can capture
