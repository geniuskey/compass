---
title: Energy Budget Analyzer
---

# Energy Budget Analyzer

Track where every photon's energy goes as it traverses the pixel stack. At each wavelength, the incident energy is split into reflection, absorption in each layer, and transmission.

<EnergyBudgetAnalyzer />

## Understanding the Energy Budget

The fundamental constraint is energy conservation: **R + A + T = 1**

- **Reflection (R)** — light bounced back at interfaces, especially at the air-microlens and BARL-silicon boundaries
- **Silicon absorption** — the useful signal; this equals the quantum efficiency
- **Color filter absorption** — intentional wavelength-selective absorption
- **BARL/planarization/microlens absorption** — parasitic losses (ideally zero)
- **Transmission (T)** — light that passes through silicon without being absorbed, significant for red/NIR wavelengths in thin silicon

::: info
Blue photons (λ ≈ 450nm) are absorbed within the first 0.5μm of silicon, while red photons (λ ≈ 650nm) require 2-3μm for full absorption.
:::
