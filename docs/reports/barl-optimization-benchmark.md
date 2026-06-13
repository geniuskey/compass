---
outline: deep
---

# BARL Optimization Benchmark

_Generated on 2026-06-11 with the COMPASS TMM solver on the generic 1.0 um BSI stack._

This report turns the BARL cookbook recipe into a reproducible benchmark. It sweeps single-layer Si3N4 thickness, a two-layer SiO2/HfO2 grid, and then compares representative designs on the same 400-700 nm wavelength grid.

## Executive summary

- The 550 nm ideal ARC index between `cf_green` and silicon is 2.5148; Si3N4 gives a quarter-wave thickness of 67.9612 nm.
- The best single-layer Si3N4 sweep point is 70 nm with mean R=0.0604.
- The best two-layer SiO2/HfO2 grid point is SiO2 5 nm / HfO2 65 nm with mean R=0.0627.
- Best design in this candidate set reduces mean reflection by 0.029 absolute versus no BARL.

::: warning Planar proxy
This is a TMM planar green-stack benchmark. It is appropriate for BARL screening and reflection trends, but it does not include lateral Bayer geometry, microlens focusing, metal-grid diffraction, or crosstalk.
:::

## Candidate Spectra

![BARL design spectra](/reports/barl-optimization/01_barl_design_spectra.png)

## Single-layer Sweep

![Single-layer BARL sweep](/reports/barl-optimization/02_single_layer_sweep.png)

## Two-layer Sweep

![Two-layer BARL heatmap](/reports/barl-optimization/03_hfo2_sio2_heatmap.png)

## Design Scorecard

![BARL scorecard](/reports/barl-optimization/04_barl_design_scorecard.png)

| Design | role | layers | total nm | mean R | max R | R@550 | mean A | energy residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No BARL | baseline | none | 0 | 0.0895 | 0.3656 | 0.0091 | 0.6492 | 1.11e-16 |
| Default 4-layer | sample default | sio2 10 nm / hfo2 25 nm / sio2 15 nm / si3n4 30 nm | 80 | 0.0698 | 0.2213 | 0.2213 | 0.6345 | 1.11e-16 |
| Si3N4 quarter-wave | analytic | si3n4 68 nm | 67.9612 | 0.0605 | 0.1788 | 0.1788 | 0.6274 | 1.11e-16 |
| Best single Si3N4 | sweep best | si3n4 70 nm | 70 | 0.0604 | 0.1766 | 0.1766 | 0.6272 | 0 |
| Best SiO2/HfO2 | sweep best | sio2 5 nm / hfo2 65 nm | 70 | 0.0627 | 0.195 | 0.195 | 0.6294 | 1.11e-16 |

## Interpretation

- BARL tuning should optimize broadband reflection, not only R at 550 nm. A quarter-wave layer is a useful seed, but it is not automatically the broadband optimum once color-filter and planarization phases are included.
- The default sample BARL is an illustrative process stack, not a guaranteed optimum. This report makes that explicit by comparing it against simple swept alternatives.
- The two-layer optimum sits on the lower SiO2 sweep boundary, so the next local search should extend toward thinner SiO2 or test HfO2-only variants.
- After a TMM BARL candidate is selected, run a patterned RCWA check because the metal grid and microlens can move the apparent optimum.

## Regeneration

```powershell
uv run python scripts\generate_barl_optimization_report.py
```

Generated metrics are stored at `docs/public/reports/barl-optimization/barl_optimization_metrics.json`.
