---
outline: deep
---

# Simulation Reports

Publication-style reports generated from Python benchmark artifacts and geometry audit scripts. Reports are for validation evidence: generated figures, metric tables, and exact regeneration commands.

## Available reports

- [RCWA/FDTD Convergence Analysis](./convergence-analysis.md) (generated 2026-05-07)
- [Pixel Stack Geometry Audit](./pixel-stack-geometry-audit.md) (generated 2026-05-08)
- [Color Filter Relief Sensitivity](./color-filter-relief-sensitivity.md) (generated 2026-05-08)
- [Guide Interactive Component Audit](./guide-interactive-audit.md) (generated 2026-05-13)

## Report queue

| Priority | Report | Evidence required |
| --- | --- | --- |
| 1 | RCWA backend parity | torcwa/grcwa/meent/fmmax QE, R/T/A, runtime table |
| 2 | Angular response characterization | Structured $(\theta,\phi,\lambda)$ QE/EQE grid, ray-file cone averaging, CRA/F-number/corner sampling maps |
| 3 | BARL optimization benchmark | Thickness/material sweep promoted from local outputs |
| 4 | DTI crosstalk benchmark | FDTI/BDTI width/depth/material sweep with crosstalk matrix |
| 5 | Performance benchmark | CPU/GPU runtime, memory, wavelength-sweep cost |

## Characterization report template

The next angular-response report should separate the evidence into three layers:

| Layer | Required output | Why it matters |
| --- | --- | --- |
| Optical angular grid | $\text{QE}(\lambda,\theta,\phi)$ or $\text{OE}(\lambda,\theta,\phi)$ per pixel | Reusable lookup table for many lens positions |
| Cone/ray averaging | Weighted averages using `intensity * weight` for each ray bundle | Converts angular response into sensor-position response |
| Electrical collection, when available | $W_i(\mathbf{r})$ map or documented approximation | Distinguishes optical absorption from collected charge |

Do not publish a single cone-averaged curve without the angular grid and weighting convention used to produce it. Otherwise the result is difficult to compare across solvers, lens files, or sensor positions.

External workflow reference: Ansys Optics' [CMOS Sensor Camera - Sensor Characterization](https://optics.ansys.com/hc/en-us/articles/360062131614-CMOS-Sensor-Camera-Sensor-Characterization) is a useful example of this separation between angular optical simulation, electrical weighting, and ray-based cone averaging. COMPASS should reuse the workflow concepts without depending on a specific commercial tool chain.

## What belongs here

- Cross-solver validation results that should be inspectable from GitHub Pages.
- Geometry audits that prove the solver input stack matches the intended config.
- Plots and tables promoted from local `outputs/` artifacts into `docs/public/reports/`.
- Reproducibility notes that explain which scripts regenerate the published figures.

Use [Theory](/theory/) for concepts, [Guide](/guide/) for workflows, [Cookbook](/cookbook/bsi-2x2-basic) for recipes, and Reports for generated evidence.
