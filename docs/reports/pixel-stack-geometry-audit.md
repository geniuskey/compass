---
outline: deep
---

# Pixel Stack Geometry Audit

_Generated on 2026-05-08 from `compass.geometry.sample_pixels` and `PixelStack`._

This report publishes geometry evidence, not optical performance. It verifies that representative sample-pixel presets expand into plausible solver input stacks with color-filter relief, metal-grid thickness, DTI, microlens, and photodiode windows present in the generated `PixelStack`.

## Executive summary

- All audited presets produce color-filter relief slices rather than a single flat slab.
- The color-filter stack height covers the tallest RGB channel and the metal grid thickness for every audited preset.
- Photodiode x-y windows stay inside the pixel pitch for every audited preset.
- This report does not claim QE or crosstalk deltas. Use it before optical sweeps to confirm the geometry being simulated.

## Geometry overview

![PixelStack geometry overview](/reports/geometry/pixel-stack-audit/sample_stack_overview.png)

## Audited presets

| Preset | pitch um | unit cell | CF stack um | grid um | R/G/B CF um | min angle | CF slices | PD xy fill |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generic 1.0 um BSI | 1 | 2x2 | 0.648 | 0.468 | 0.624/0.6/0.648 | 62 | 11 | 0.49 |
| 0.56 um 4x4 OCL | 0.56 | 8x8 | 0.594 | 0.429 | 0.572/0.55/0.594 | 62 | 11 | 0.49 |
| 1.0 um Quad Bayer | 1 | 4x4 | 0.648 | 0.468 | 0.624/0.6/0.648 | 62 | 11 | 0.49 |
| 1.22 um 2x2 OCL | 1.22 | 4x4 | 0.756 | 0.546 | 0.728/0.7/0.756 | 62 | 11 | 0.49 |
| 1.6 um split PD | 1.6 | 2x2 | 0.94 | 0.679 | 0.905/0.87/0.94 | 62 | 11 | 0.774 |
| 1.2 um LOFIC | 1.2 | 4x4 | 0.745 | 0.538 | 0.718/0.69/0.745 | 62 | 11 | 0.423 |

## Checks

| Preset | CF covers channels | grid <= stack | multi-slice relief | PD inside pixel |
| --- | --- | --- | --- | --- |
| Generic 1.0 um BSI | yes | yes | yes | yes |
| 0.56 um 4x4 OCL | yes | yes | yes | yes |
| 1.0 um Quad Bayer | yes | yes | yes | yes |
| 1.22 um 2x2 OCL | yes | yes | yes | yes |
| 1.6 um split PD | yes | yes | yes | yes |
| 1.2 um LOFIC | yes | yes | yes | yes |

## Regeneration

```powershell
uv run python scripts\generate_geometry_reports.py
```

Generated metrics are stored at `docs/public/reports/geometry/pixel-stack-audit/geometry_metrics.json`.
