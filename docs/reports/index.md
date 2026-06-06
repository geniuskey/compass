---
outline: deep
---

# Simulation Reports

Publication-style reports generated from Python benchmark artifacts and geometry audit scripts. Reports are for validation evidence: generated figures, metric tables, and exact regeneration commands.

## Available reports

- [RCWA/FDTD Convergence Analysis](./convergence-analysis.md) (generated 2026-05-07)
- [Pixel Stack Geometry Audit](./pixel-stack-geometry-audit.md) (generated 2026-06-06)
- [Color Filter Relief Sensitivity](./color-filter-relief-sensitivity.md) (generated 2026-06-06)
- [Pixel Structure Realism](./pixel-structure-realism.md) (generated 2026-06-06)

## Report queue

| Priority | Report | Evidence required |
| --- | --- | --- |
| 1 | RCWA backend parity | torcwa/grcwa/meent/fmmax QE, R/T/A, runtime table |
| 2 | CRA cone illumination sweep | CRA/F-number/corner sampling maps and convergence table |
| 3 | BARL optimization benchmark | Thickness/material sweep promoted from local outputs |
| 4 | DTI crosstalk benchmark | FDTI/BDTI width/depth/material sweep with crosstalk matrix |
| 5 | Performance benchmark | CPU/GPU runtime, memory, wavelength-sweep cost |

## What belongs here

- Cross-solver validation results that should be inspectable from GitHub Pages.
- Geometry audits that prove the solver input stack matches the intended config.
- Plots and tables promoted from local `outputs/` artifacts into `docs/public/reports/`.
- Reproducibility notes that explain which scripts regenerate the published figures.

Use [Theory](/theory/) for concepts, [Guide](/guide/) for workflows, [Cookbook](/cookbook/bsi-2x2-basic) for recipes, and Reports for generated evidence.
