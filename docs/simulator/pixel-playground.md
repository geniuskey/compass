---
title: Pixel Design Playground
---

# Pixel Design Playground

The comprehensive pixel design tool. Configure every parameter of a BSI CMOS pixel — from silicon thickness to BARL sublayers — and see the combined effect on QE, reflectance, energy budget, and stack geometry.

<PixelDesignPlayground />

## Design Guidelines

### Maximizing QE
1. **Thick silicon** (3-4 μm) for red/NIR absorption
2. **Optimized BARL** to minimize interface reflection
3. **Thin color filter** to reduce parasitic absorption outside the passband
4. **Low-loss planarization** layer

### Minimizing Crosstalk
1. Keep color filter absorption high (thick CF with sharp spectral edges)
2. Reduce silicon-transmitted light (relevant for thin Si at long wavelengths)

::: tip
Start with a preset and adjust one parameter at a time to understand its effect. Compare the "BSI 1μm" and "High QE" presets to see the impact of silicon thickness and BARL optimization.
:::
