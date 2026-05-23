---
title: Microlens Process Shape Predictor
---

# Microlens Process Shape Predictor

Estimate how lithographic layout, thermal reflow, and etch-transfer settings affect final CIS microlens gap, height, curvature, and 3D profile.

<MicrolensProcessShape />

::: info Model scope
This browser tool is a **surrogate process model**, not a calibrated foundry recipe. Use it to reason about process direction, sensitivity, and failure modes. Replace the normalized coefficients with wafer metrology data before using it for quantitative process decisions.
:::

## What This Adds

Most existing COMPASS microlens tools start from an assumed lens geometry. This tool starts one step earlier:

- **Layout**: pixel pitch, lithographic island width, aperture footprint shape, and a lens-unit layout selector for 1x1, 2x1, 1x2, 2x2 on-chip-lens groups (presets plus a custom 4x4 editor for heterogeneous die-level mixing of, e.g., 2x2 OCL with 1x1 fill).
- **Reflow**: temperature/time as a normalized thermal budget that spreads the lens and changes height through volume conservation. Asymmetric (2x1 / 1x2) masks include a surface-tension correction that drives the long axis to spread less and the short axis to spread more, consistent with reflowed-photoresist observations.
- **Etch transfer**: etch time closes residual lens gap; polymerization preserves height; mask thickness changes transfer robustness.
- **Calibration**: reflow spread, volume retention, lateral etch, and vertical height-loss gains can be fitted to AFM/SEM or DOE data.
- **Outputs**: final gap (worst of X/Y), height, aspect ratio (WX:WY), vertex radius of curvature, f-number, fill factor, zero-gap etch-time estimate, profile exponent, cross-section, **top-down (XY) footprint view with height heat-map for each lens group**, 3D wireframe, and etch response curves.

## Literature Basis

- Ristoiu et al., **A DOE study of plasma etched microlens shape for CMOS image sensors**, SPIE 2020. This is the closest direct CIS process reference: reflowed microlenses are plasma transferred, and gap/height evolution is modeled versus mask thickness, polymerizing gas flow, and etch time. DOI: [10.1117/12.2551857](https://doi.org/10.1117/12.2551857)
- Baillie and Gendler, **Zero-space microlenses for CMOS image sensors: optical modeling and lithographic process development**, SPIE 2004. This motivates the layout-gap and zero-space problem: too little lithographic space risks merging during melt/reflow, while residual space reduces fill factor. DOI: [10.1117/12.533453](https://doi.org/10.1117/12.533453)
- Jin, Liu, and Yang, **Design, characterization and evaluation of high performance 2.8 μm pitch zero space microlens**, *Optics Communications*, 2011. This connects zero-space microlens geometry to AFM characterization and silicon-level sensitivity/crosstalk tests. DOI: [10.1016/j.optcom.2010.11.073](https://doi.org/10.1016/j.optcom.2010.11.073)
- Tan, Goh, and Kim, **Microfabrication of Microlens by Timed-Development-and-Thermal-Reflow**, *Micromachines*, 2020. This supports the parabolic/superellipse profile framing and shows how aperture geometry, development time, diameter, thickness, radius of curvature, and focal length can be regression-modeled. DOI: [10.3390/mi11030277](https://doi.org/10.3390/mi11030277)
- Choi *et al.*, **Profile control of asymmetric reflowed microlens for CMOS image sensors**, *Microelectronic Engineering* (2014). This is the basis for the long-axis-shrinks / short-axis-grows surface-tension correction used here for 2x1 and 1x2 lens units: asymmetric reflowed photoresist evolves toward an isotropic equilibrium shape, so the post-reflow X gap and Y gap end up unequal even when the as-printed boundary gap is identical.
- Y. Oike *et al.* (Sony Semiconductor Solutions), **All-pixel phase-detection autofocus pixel architectures with 2x1 on-chip lens (Dual Pixel CMOS Image Sensor)**, ISSCC / IEEE Journal of Solid-State Circuits. The 2x1 OCL footprint shares one shaped lens across two adjacent photodiodes; the lens long axis is the PDAF separation direction. Motivates the **All 2x1 (Sony 2PD)** preset.
- J. Park *et al.* (Samsung), **A 1/2.55-inch 1.0 μm-pixel 64-Mpixel CMOS image sensor with Tetracell color filter array**, ISSCC. Same-color 2x2 sub-pixels share one 2x2 OCL for low-light binning; the boundary against adjacent (different-color) 2x2 cells is the merger-risk surface. Motivates the **All 2x2 (Tetracell OCL)** preset.

::: warning
The public papers do not expose a universal CIS recipe. The simulator therefore preserves the **directional relationships** from the literature and makes the coefficients explicit enough to replace later with DOE or AFM/SEM data.
:::

::: tip Related tools
[Microlens Ray Trace](./microlens-raytrace) · [MLA Array Visualizer](./mla-array) · [Microlens & CRA recipe](/cookbook/microlens-optimization)
:::

<SimulatorTheory slug="microlens-process-shape" />
