---
title: Pixel Structure Realism
description: What a physically realistic BSI CMOS image sensor pixel contains beyond an idealised layer stack — backside inverted-pyramid light-trapping texture, conformal high-k DTI liners, tapered trench profiles, and the microlens residual base — and how COMPASS models each one.
---

# Pixel Structure Realism

::: tip Prerequisites
[Pixel Anatomy](../basics/pixel-anatomy.md) introduces the layer stack and
[Pixel Optical Effects](./pixel-optical-effects.md) explains how each component
moves QE and crosstalk. This page is about the *structural* details that
separate a textbook stack from a real fabricated pixel.
:::

A first-order pixel model — flat silicon, a rectangular oxide-filled trench, a
clean microlens cap — is enough to reproduce the dominant QE trends. But the
last few percent of QE, the near-infrared (NIR) response, and the crosstalk
floor are all set by structural details that a flat model misses. This page
collects the realism features that matter and maps each to its COMPASS config
knob.

## Realism gap map

| Real-pixel feature | Why it matters optically | COMPASS config |
|---|---|---|
| Backside inverted-pyramid texture | Graded-index moth-eye AR + light trapping; large NIR QE gain | `silicon.surface_texture` |
| DTI conformal high-k liner | Thin high-index ring; passivation + negative fixed charge | `silicon.dti.liner` |
| Tapered DTI sidewall | Etched trenches narrow with depth; changes deep isolation fill | `silicon.dti.taper_angle` |
| Microlens residual base | Reflow/etch-back leaves a flat polymer slab under the cap | `microlens.base_thickness` |
| Per-color CF relief + contact angle | Each resist has its own height and sidewall slope | `color_filter.<color>` |
| Metal grid with rounded corners | Real grids are not perfect squares | `color_filter.grid.corner_radius` |

## Backside inverted-pyramid texture (light trapping)

Modern NIR-enhanced BSI sensors etch an **inverted-pyramid array (IPA)** into
the silicon backside that faces the incoming light. Two mechanisms boost QE:

1. **Graded-index anti-reflection.** Going from the trench fill into bulk
   silicon, the area-averaged silicon fraction rises smoothly from ~0 at the
   surface to 1 at the pyramid apex. The effective index therefore ramps
   gradually rather than stepping abruptly at a flat Si interface, suppressing
   the front-surface reflection in the same way a moth-eye coating does.
2. **Light trapping.** The faceted surface refracts long-wavelength light to
   oblique angles, lengthening its path in the (weakly absorbing) silicon and
   increasing the chance of absorption before it escapes.

Published device simulations and process reports show that combining an IPA
with deep DTI raises near-infrared QE substantially — on the order of ~3x at
850 nm and ~5x at 940 nm relative to a planar backside — because silicon's
absorption length grows to tens of microns in that band.

COMPASS models the IPA as a staircase of pyramidal pits carved from the top of
the silicon layer and back-filled with `fill_material`. The pit half-width
shrinks linearly from `period/2` at the surface to zero at the apex, producing
the graded silicon fill fraction:

```yaml
silicon:
  surface_texture:
    enabled: true
    type: inverted_pyramid
    height: 0.35          # texture depth into silicon (um)
    period: null          # defaults to the pixel pitch
    fill_material: sio2
    n_slices: 8           # staircase resolution for RCWA
```

## DTI: liner, fill, and taper

A real backside deep trench is not a clean oxide rectangle:

- **Conformal high-k liner.** The etched silicon sidewall is lined with a thin
  high-k film (Al2O3, HfO2, or Ta2O5, typically tens of nm) that both
  passivates dangling bonds and carries a **negative fixed charge** to repel
  electrons from the damaged surface, lowering dark current. Optically it is a
  thin high-index ring sitting between the silicon and the lower-index core
  fill, so it changes the trench's reflection and its crosstalk contribution.
- **Core fill.** Inside the liner the trench is filled with oxide (most common),
  or in some designs poly-silicon, tungsten (metal DTI), or even an air gap.
- **Tapered profile.** Plasma-etched trenches narrow with depth. A vertical-wall
  idealisation over-counts the isolation material deep in the substrate and
  mislocates the trench/silicon boundary for ray and field calculations.

```yaml
silicon:
  dti:
    enabled: true
    mode: fdti
    width: 0.12           # trench width at the backside opening (um)
    depth: 4.0
    material: sio2        # core fill
    taper_angle: 82.0     # sidewall angle from substrate plane (90 = vertical)
    n_slices: 6           # staircase resolution for the taper
    liner:
      enabled: true
      material: al2o3     # high-k passivation liner
      thickness: 0.015
```

When the liner is disabled, the taper angle is 90°, and no texture is present,
COMPASS uses its original fast single-slice (FDTI) or two-slice (BDTI) path, so
existing configs are unchanged. Enabling any of these features switches the
silicon layer to a z-resolved staircase that the RCWA and FDTD back-ends consume
transparently.

## Microlens residual base

Polymer microlenses are formed by reflowing or etching back a patterned resist.
The process always leaves a **flat residual slab** of the same polymer beneath
the curved cap — the lens is never zero-thickness at its edges. Ignoring it
slightly under-estimates the optical path in the lens material and the height of
the air gap above the color filter.

```yaml
microlens:
  height: 0.67            # curved cap sag
  base_thickness: 0.10    # flat residual polymer slab under the cap
```

## Putting it together

The `sample_p1p12um_nir` preset turns all of these on at once and is the subject
of the [Pixel Structure Realism report](/reports/pixel-structure-realism),
which renders the actual solver permittivity `Re(eps)(x, z)` so you can see the
graded texture, the lined tapered trench, and the residual lens base in the
geometry the solver integrates.

```bash
python scripts/run_simulation.py pixel=sample_p1p12um_nir
```

::: warning Scope
These features make the *geometry* more faithful. Whether they change QE for
your stack is an optical question — run an RCWA order-converged sweep (and, for
the texture, a wavelength sweep into the NIR) before quoting deltas.
:::
