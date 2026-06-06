---
outline: deep
---

# Color Filter Relief Sensitivity Report

_Generated on 2026-06-06 from the generic 1.0 um BSI `PixelStack`._

This is a geometry-sensitivity report for the per-channel color-filter model. It shows how `grid.thickness`, `red/green/blue.thickness`, and `red/green/blue.contact_angle` change the z-sliced solver geometry.

::: info Scope
The figures below are geometry evidence. They do not yet report optical QE or crosstalk deltas. The next optical report should run RCWA order sweeps over these geometry variants.
:::

## Cross-section variants

![Color filter relief cross sections](/reports/geometry/color-filter-relief/color_filter_relief_sections.png)

## Contact-angle sweep

![Contact angle sweep](/reports/geometry/color-filter-relief/contact_angle_sweep.png)

## Default per-channel geometry

| Color | material | thickness um | above grid um | contact angle | top area / pitch area |
| --- | --- | --- | --- | --- | --- |
| R | cf_red | 0.624 | 0.156 | 66 | 0.61 |
| G | cf_green | 0.6 | 0.132 | 72 | 0.696 |
| B | cf_blue | 0.648 | 0.18 | 62 | 0.531 |

## Interpretation

- `grid.thickness` defines the vertical part of the metal-grid region.
- Channel `thickness` values define the maximum color-resist height per color.
- `contact_angle` controls the trapezoidal taper above the grid. Lower angle means a smaller top footprint for the same protrusion height.
- Because red, green, and blue use different heights and angles, RCWA receives multiple color-filter z slices even before microlens slicing is considered.

## Regeneration

```powershell
uv run python scripts\generate_geometry_reports.py
```

Generated metrics are stored at `docs/public/reports/geometry/color-filter-relief/color_filter_relief_metrics.json`.
