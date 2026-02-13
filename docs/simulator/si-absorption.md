---
title: Silicon Absorption Depth
---

# Silicon Absorption Depth

Visualize how photon absorption depth in silicon varies with wavelength using the Beer-Lambert law. Understand why pixel thickness is a critical design parameter for quantum efficiency.

<SiliconAbsorptionDepth />

## Physics

The intensity of light propagating through silicon decays exponentially according to the Beer-Lambert law:

**I(z) = I₀ exp(-αz)**

where **α = 4πk/λ** is the absorption coefficient, computed from the imaginary part of silicon's complex refractive index (n + ik).

### Wavelength Dependence

- **Blue (400-500 nm):** α is very large, so photons are absorbed within the first ~0.5 um. Even thin silicon captures blue light efficiently.
- **Green (500-600 nm):** Moderate absorption depth (~1-2 um). Standard BSI pixel thicknesses capture most green photons.
- **Red/NIR (600-1000 nm):** α drops steeply near the silicon bandgap (1.12 eV, ~1100 nm). Absorption depths exceed 5-10 um, making thicker silicon essential for red QE.

::: tip Design Implication
Increasing silicon thickness from 3 um to 6 um has negligible impact on blue QE but can improve NIR QE by 2-3x. This is a key motivation for deep-trench BSI architectures.
:::
