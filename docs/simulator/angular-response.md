---
title: Angular Response
---

# Angular Response Simulator

In real cameras, light hits pixels at various angles depending on the pixel's position on the sensor and the lens design. The Chief Ray Angle (CRA) can reach 20-30° at the sensor edge, significantly affecting QE.

<AngularResponseSimulator />

## CRA Effects

As the angle of incidence increases:
- **Reflectance increases** due to Fresnel effects (especially for s-polarization)
- **Effective path length increases** through each layer, shifting thin-film interference conditions
- **QE generally decreases**, with the rate depending on the BARL design and color filter

::: tip
A well-designed BARL maintains >80% of normal-incidence QE up to 15° CRA. Beyond 20°, QE degradation becomes significant for most pixel designs.
:::
