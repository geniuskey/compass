---
outline: deep
---

# Pixel Structure Realism Report

_Generated on 2026-06-06 from `compass.geometry.sample_pixels` and `PixelStack`._

This report documents the structural-realism features COMPASS models for modern backside-illuminated (BSI) CMOS image-sensor pixels, and visualises them directly from the generated solver input. The cross-sections sample the real permittivity `Re(eps)(x, z)` the RCWA solver integrates, so this is geometry evidence rather than an optical-performance claim.

## Why these features matter

- **Backside inverted-pyramid texture (IPA).** A graded silicon fill fraction at the backside acts as a moth-eye anti-reflection / light-trapping layer. Published simulations and process reports show large near-infrared QE gains (up to ~3x at 850 nm and ~5x at 940 nm) when an IPA is combined with deep DTI.
- **DTI conformal high-k liner.** Real backside DTI trenches are lined with a thin high-k film (Al2O3 / HfO2 / Ta2O5, ~30-100 nm) that passivates the etched silicon and carries a negative fixed charge. Optically it is a thin high-index ring between silicon and the lower-index trench fill.
- **Tapered DTI sidewall.** Etched trenches narrow with depth; a vertical-wall idealisation over-counts the isolation oxide deep in the substrate.
- **Microlens residual base.** Reflow / etch-back leaves a flat polymer slab under the curved cap; the lens is never zero-thickness at its edges.

## Baseline vs NIR-enhanced pixel

![Baseline vs realism-enhanced pixel](/reports/geometry/structure-realism/baseline_vs_realism.png)

## Silicon backside detail

The graded `Re(eps)` of the inverted-pyramid texture produces a smooth effective-index transition from the trench fill toward bulk silicon, while the tapered DTI trench and its high-k liner are visible at the pixel boundary.

![Silicon backside detail](/reports/geometry/structure-realism/silicon_backside_zoom.png)

- Texture height: **0.350 um**
- Area-averaged effective index across the texture: **1.46** at the surface to **3.67** toward the apex (monotonic graded-index anti-reflection).

## Capability coverage matrix

| Real-pixel feature | COMPASS status | config key |
| --- | --- | --- |
| Microlens superellipse profile + CRA shift | modelled | `microlens.profile, microlens.shift` |
| Multi-pixel shared lens (2x2 / 4x4 OCL) | modelled | `microlens.sharing` |
| Microlens residual base layer | modelled (new) | `microlens.base_thickness` |
| Per-color CF thickness + contact-angle relief | modelled | `color_filter.<color>.thickness/contact_angle` |
| Metal grid (W) with rounded corners | modelled | `color_filter.grid` |
| BARL anti-reflection multilayer | modelled | `barl.layers` |
| FDTI / BDTI deep trench isolation | modelled | `silicon.dti.mode/depth/width` |
| DTI conformal high-k passivation liner | modelled (new) | `silicon.dti.liner` |
| Tapered DTI sidewall (etch profile) | modelled (new) | `silicon.dti.taper_angle` |
| Backside inverted-pyramid light-trapping texture | modelled (new) | `silicon.surface_texture` |
| Photodiode collection window | modelled | `silicon.photodiode` |
| Composite / air-gap metal grid liner | roadmap | `-` |
| In-pixel light pipe / inner lens | roadmap | `-` |

## Reproduce this pixel

```bash
python scripts/run_simulation.py pixel=sample_p1p12um_nir
```

Or derive a config in Python:

```python
from compass.geometry.sample_pixels import derive_parameters
from compass.geometry.pixel_stack import PixelStack

cfg = derive_parameters("sample_p1p12um_nir")
stack = PixelStack({"pixel": cfg})
```

## Regeneration

```bash
python scripts/generate_realism_report.py
```

Generated metrics are stored at `docs/public/reports/geometry/structure-realism/structure_realism_metrics.json`.
