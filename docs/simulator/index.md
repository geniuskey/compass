---
title: Simulator
---

# Interactive Simulators

Explore CMOS image sensor pixel optics through browser-based simulators. Calculations run locally in the browser — no server required.

::: tip Where these fit
The simulators are **lightweight pedagogical tools**: TMM, geometric optics, analytical camera equations, and visualizers. They are for intuition, classroom use, and design-space scoping. For physically accurate per-pixel QE / crosstalk / field maps, run the [Python solver pipeline](/guide/quickstart) (RCWA/FDTD/Meep). See the scope table at the bottom for what each tool family does and does not cover.
:::

## Available Simulators

Pick a tool by category, or filter by keyword.

<SimulatorGallery />

## Quick Validation Examples

Use these checks before trusting a simulator trend. They are intentionally simple: a good browser model should pass the limiting case before you interpret a complicated sweep.

| Simulator | Sanity check | Expected behavior |
|-----------|--------------|-------------------|
| [TMM QE Calculator](./tmm-qe) | Remove coating layers and compare normal-incidence air/Si reflection with $R=|(1-n_{Si})/(1+n_{Si})|^2$. | Reflectance should be close to the Fresnel limit and energy should satisfy $R+T+\sum A_j\approx1$. |
| [Microlens Process Shape](./microlens-process-shape) | Hold reflow fixed and increase etch time. | Final gap should decrease while height retention decreases; excessive etch should trigger a height-loss or over-etch warning. |
| [Color Filter Designer](./color-filter) | Narrow RGB filter FWHM while keeping peaks fixed. | Channel crosstalk should drop, but total signal and low-light SNR margin should also shrink. |
| [SNR Calculator](./snr-calculator) / [Photon Transfer Curve](./photon-transfer-curve) | Sweep illumination with read noise fixed. | Low signal should be read-noise limited; high signal should follow shot-noise behavior, $\sigma\approx\sqrt{S}$. |
| [Si Absorption Depth](./si-absorption) | Increase silicon thickness at red/NIR wavelengths. | Red/NIR absorption should improve more strongly than blue absorption because long wavelengths penetrate deeper. |

## Model Scope

| Tool family | Model type | Best for |
|-------------|------------|----------|
| TMM / thin-film tools | 1D planar multilayer optics | BARL trends, reflectance, silicon absorption, first-pass QE |
| Ray / geometry tools | Paraxial or Snell-law geometric optics | Microlens intuition, CRA directionality, layout sensitivity |
| Sensor metrics tools | Analytical camera equations | SNR, dynamic range, MTF, EMVA-style comparisons |
| Visualizers | Deterministic illustrative models | Explaining structure, relative trends, and failure modes |

::: warning
Treat these as educational and design-space tools. They do not replace RCWA/FDTD sign-off, silicon calibration, or vendor process data. For solver workflows, see the [Guide](/guide/) and [Reports](/reports/).
:::
