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

- **Layout**: pixel pitch, lithographic island width, and aperture footprint shape.
- **Reflow**: temperature/time as a normalized thermal budget that spreads the lens and changes height through volume conservation.
- **Etch transfer**: etch time closes residual lens gap; polymerization preserves height; mask thickness changes transfer robustness.
- **Outputs**: final gap, height, vertex radius of curvature, f-number, fill factor, cross-section, 3D wireframe, and etch response curves.

## Literature Basis

- Ristoiu et al., **A DOE study of plasma etched microlens shape for CMOS image sensors**, SPIE 2020. This is the closest direct CIS process reference: reflowed microlenses are plasma transferred, and gap/height evolution is modeled versus mask thickness, polymerizing gas flow, and etch time. DOI: [10.1117/12.2551857](https://doi.org/10.1117/12.2551857)
- Baillie and Gendler, **Zero-space microlenses for CMOS image sensors: optical modeling and lithographic process development**, SPIE 2004. This motivates the layout-gap and zero-space problem: too little lithographic space risks merging during melt/reflow, while residual space reduces fill factor. DOI: [10.1117/12.533453](https://doi.org/10.1117/12.533453)
- Jin, Liu, and Yang, **Design, characterization and evaluation of high performance 2.8 μm pitch zero space microlens**, *Optics Communications*, 2011. This connects zero-space microlens geometry to AFM characterization and silicon-level sensitivity/crosstalk tests. DOI: [10.1016/j.optcom.2010.11.073](https://doi.org/10.1016/j.optcom.2010.11.073)
- Tan, Goh, and Kim, **Microfabrication of Microlens by Timed-Development-and-Thermal-Reflow**, *Micromachines*, 2020. This supports the parabolic/superellipse profile framing and shows how aperture geometry, development time, diameter, thickness, radius of curvature, and focal length can be regression-modeled. DOI: [10.3390/mi11030277](https://doi.org/10.3390/mi11030277)

::: warning
The public papers do not expose a universal CIS recipe. The simulator therefore preserves the **directional relationships** from the literature and makes the coefficients explicit enough to replace later with DOE or AFM/SEM data.
:::

::: tip Related tools
[Microlens Ray Trace](./microlens-raytrace) · [MLA Array Visualizer](./mla-array) · [Microlens & CRA recipe](/cookbook/microlens-optimization)
:::
