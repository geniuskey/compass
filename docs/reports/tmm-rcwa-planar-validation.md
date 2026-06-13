---
outline: deep
---

# TMM vs RCWA Planar Stack Validation

_Generated on 2026-06-11 from `transfer_matrix_1d` and direct `torcwa` zero-order RCWA solves._

This report isolates the planar-stack limit. With no lateral patterning, zero-order RCWA should reduce to the same 1D optics solved by TMM. That makes this the first validation rung before using RCWA on Bayer patterns, metal grids, DTI trenches, or microlenses.

## Executive summary

- All four planar validation cases pass the 5e-5 R/T/A agreement target; the worst spectral difference is **2.59e-06**.
- The ideal quarter-wave ARC reduces the 550 nm bare silicon reflectance from **0.36** to **2.11e-33** in TMM and **4.63e-15** in RCWA.
- This report is normal-incidence and planar-only. It intentionally does not validate lateral diffraction, color-filter relief, DTI crosstalk, or photodiode collection.

## R/T/A alignment

![TMM vs RCWA RTA alignment](/reports/tmm-rcwa-planar/01_rta_alignment.png)

## Error summary

![TMM vs RCWA error summary](/reports/tmm-rcwa-planar/02_error_summary.png)

## Quarter-wave ARC sanity check

![Quarter-wave ARC reflectance](/reports/tmm-rcwa-planar/03_arc_reflectance.png)

## Validation table

| Case | layers | thickness um | max \|dR\| | max \|dT\| | max \|dA\| | RCWA energy residual | passes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Air / glass interface | 0 | 0 | 2.83e-09 | 9.78e-08 | 1.01e-07 | 0 | yes |
| Ideal quarter-wave ARC on silicon | 1 | 0.0688 | 6.15e-08 | 6.85e-07 | 6.93e-07 | 7.45e-08 | yes |
| Lossless pixel-like multilayer | 7 | 0.99 | 1.14e-06 | 1.62e-06 | 2.59e-06 | 4.47e-08 | yes |
| Lossy pixel-like multilayer | 7 | 0.99 | 7.26e-07 | 1.06e-06 | 1.50e-06 | 0 | yes |

## Interpretation

- The single-interface row checks Fresnel normalization without any finite films.
- The ARC row verifies interference phase and the expected quarter-wave reflectance null.
- The lossless multilayer checks phase accumulation through a pixel-like dielectric ladder.
- The lossy multilayer checks that complex refractive indices and absorption accounting are aligned.

## Regeneration

```powershell
uv run python scripts\generate_tmm_rcwa_planar_report.py
```

Generated metrics are stored at `docs/public/reports/tmm-rcwa-planar/tmm_rcwa_planar_metrics.json`.
