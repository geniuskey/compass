---
title: "Responsivity Calculator"
---

# Responsivity Calculator

Calculate spectral responsivity from TMM-based quantum efficiency. Compare per-channel responsivity against the ideal silicon photodiode response.

<ResponsivityCalculator />

## Responsivity

Spectral responsivity converts QE to an electrical output metric:

**R(λ) = QE(λ) × qλ / (hc) = QE(λ) × λ_nm / 1240 [A/W]**

Where:
- **q** — electron charge (1.602 × 10⁻¹⁹ C)
- **h** — Planck's constant (6.626 × 10⁻³⁴ J·s)
- **c** — speed of light (2.998 × 10⁸ m/s)
- **λ_nm** — wavelength in nanometers

### Per-Channel Response

Each color channel (R, G, B) has a distinct responsivity curve shaped by:
- Color filter transmittance spectrum
- Silicon absorption depth at each wavelength
- Anti-reflection coating efficiency
- Microlens collection efficiency

### Ideal Si Photodiode

The ideal response assumes QE = 1 at all wavelengths:

**R_ideal(λ) = λ_nm / 1240 [A/W]**

::: tip
Peak responsivity wavelength differs from peak QE wavelength because responsivity includes the λ/1240 factor, which shifts the peak toward longer wavelengths.
:::
