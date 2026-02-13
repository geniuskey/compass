---
title: TMM QE Calculator
---

# TMM QE Calculator

Compute the quantum efficiency spectrum of a BSI CMOS pixel using the Transfer Matrix Method. Adjust silicon thickness, BARL layers, and angle of incidence to see real-time results.

<TmmQeCalculator />

## How It Works

The TMM models each layer as a thin film with complex refractive index. Light propagation through the stack is computed via 2×2 transfer matrices, yielding reflectance (R), transmittance (T), and per-layer absorption (A). The QE equals the fraction of light absorbed in the silicon photodiode layer.

### Key Parameters
- **Silicon thickness** controls NIR sensitivity — thicker Si absorbs more long-wavelength photons
- **Angle of incidence** simulates chief ray angle (CRA) effects from the camera lens
- **Polarization** — real scenes contain unpolarized light; s/p modes show the individual components

::: info Physics Note
This is a 1D planar approximation — it does not account for microlens focusing, DTI scattering, or lateral crosstalk. For those effects, use the full RCWA/FDTD solvers.
:::
