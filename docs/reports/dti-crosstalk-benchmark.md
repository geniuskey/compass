---
outline: deep
---

# DTI Crosstalk Benchmark

_Generated on 2026-06-11 from PixelStack geometry sweeps and a representative scalar FDTD snapshot._

This report separates cheap, reproducible DTI geometry evidence from the more expensive localized-source crosstalk benchmark. It is intended as a design-space gate before running longer vector FDTD or high-order RCWA jobs.

## Executive summary

- A 100 nm full-depth DTI occupies 0.19 of the silicon volume in the generic 1.0 um 2x2 BSI PixelStack geometry.
- A 100 nm, 0.6 um BDTI occupies 0.0393 of the silicon volume because only the backside/top portion is trenched.
- The representative 44x44x118, 3500-step scalar FDTD snapshot reports max neighbor crosstalk 0.2582 for FDTI and 0.2609 for BDTI, a gap of 0.0027 absolute.
- The periodic trench RCWA/FDTD alignment snapshot remains within about 3 percentage points in R/T/A for both FDTI and BDTI.

::: warning Scope
The crosstalk matrix is a scalar FDTD visual benchmark, not a production full-vector FDTD solve. Use it to compare geometry and normalization paths; use longer vector runs for final isolation claims.
:::

## Geometry Sweep

![DTI geometry sweeps](/reports/dti-crosstalk/01_dti_geometry_sweeps.png)

### FDTI width sweep

| width nm | max XY DTI area | effective DTI volume | open Si volume |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 1 |
| 40 | 0.0784 | 0.0784 | 0.9216 |
| 60 | 0.1164 | 0.1164 | 0.8836 |
| 80 | 0.1536 | 0.1536 | 0.8464 |
| 100 | 0.19 | 0.19 | 0.81 |
| 120 | 0.2256 | 0.2256 | 0.7744 |
| 150 | 0.2775 | 0.2775 | 0.7225 |

### BDTI depth sweep

| BDTI depth um | active depth um | effective DTI volume | open Si volume |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 1 |
| 0.3 | 0.3 | 0.0197 | 0.9803 |
| 0.6 | 0.6 | 0.0393 | 0.9607 |
| 1.2 | 1.2 | 0.0786 | 0.9214 |
| 1.8 | 1.8 | 0.1179 | 0.8821 |
| 2.4 | 2.4 | 0.1572 | 0.8428 |
| 2.9 | 2.9 | 0.19 | 0.81 |

## Silicon DTI Masks

![DTI XY and XZ masks](/reports/dti-crosstalk/02_dti_xz_masks.png)

## Representative Crosstalk Snapshot

![DTI crosstalk matrices](/reports/dti-crosstalk/03_dti_crosstalk_matrices.png)

![DTI crosstalk summary](/reports/dti-crosstalk/04_dti_crosstalk_summary.png)

| mode | mean self collection | max neighbor crosstalk | mean PD signal | energy tail change |
| --- | --- | --- | --- | --- |
| fdti | 0.5831 | 0.2582 | 15320.3545 | 0.0471 |
| bdti_0p6um | 0.5827 | 0.2609 | 15249.1798 | 0.047 |

## Periodic Trench Alignment Snapshot

| mode | max abs dR | max abs dT | max abs dA | Si absorption proxy | trench field leakage |
| --- | --- | --- | --- | --- | --- |
| fdti | 0.0266 | 0.027 | 0.0278 | 0.6087 | 0.1305 |
| bdti | 0.0183 | 0.0088 | 0.019 | 0.6105 | 0.096 |

## Interpretation

- FDTI and BDTI have similar crosstalk in the representative coarse scalar snapshot because the run is primarily a path and normalization check.
- The geometry sweep still shows the intended monotonic controls: wider FDTI increases silicon trench volume, and deeper BDTI approaches FDTI.
- A production DTI report should extend this with wavelength-resolved, localized-source vector FDTD and a true width/depth/material crosstalk sweep.

## Regeneration

```powershell
uv run python scripts\generate_dti_crosstalk_report.py
```

Generated metrics are stored at `docs/public/reports/dti-crosstalk/dti_crosstalk_metrics.json`.
