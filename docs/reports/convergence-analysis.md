---
outline: deep
---

# RCWA/FDTD Convergence Analysis Report

_Generated on 2026-05-01 from local `outputs/` benchmark artifacts._

This page turns the Python-level benchmark outputs into a publication-style report. The source JSON and plots stay in `outputs/` for local iteration, while selected figures and tables are promoted to `docs/public/reports/convergence/` so the same evidence can be served by GitHub Pages.

## Current status

- The 1D ladder is aligned: torcwa RCWA matches TMM at near numerical precision, and the 1D FDTD implementation is within the sub-percent target on the lossy pixel-like multilayer.
- The 2D periodic trench benchmark is aligned for FDTI and BDTI at the current coarse settings, with maximum R/T/A differences below roughly 3 percentage points.
- The full 2x2 pixel scalar FDTD runs are visual convergence tests. The 44x44x118, 3500-step run is the current stable comparison point. The 64x64x170 and 128x128x340 runs need longer physical runtime before their crosstalk fractions should be treated as final.

::: warning Read the high-resolution FDTD rows carefully
A fixed `--fdtd-steps` value is not equivalent across grids. Finer grids use a smaller time step, so they cover less physical time unless the step count is scaled. Compare the reported `c*time` and energy-tail values, not only the grid dimensions.
:::

## Regeneration commands

```powershell
uv run python scripts\rcwa_fdtd_alignment.py --structure lossy-multilayer --outdir outputs\rcwa_fdtd_alignment_lossy
uv run python scripts\rcwa_fdtd_trench_benchmark.py --convergence --outdir outputs\rcwa_fdtd_trench_benchmark
uv run python scripts\rcwa_fdtd_pixel_benchmark.py --fdtd-steps 3500 --outdir outputs\rcwa_fdtd_pixel_benchmark_steps3500
uv run python scripts\generate_convergence_report.py
```

High-resolution pixel checks can be regenerated with:

```powershell
uv run python scripts\rcwa_fdtd_pixel_benchmark.py --nx 64 --ny 64 --nz 170 --fdtd-steps 5200 --outdir outputs\rcwa_fdtd_pixel_benchmark_64x64x170_steps5200
uv run python scripts\rcwa_fdtd_pixel_benchmark.py --nx 128 --ny 128 --nz 340 --source-set single --fdtd-steps 10400 --outdir outputs\rcwa_fdtd_pixel_benchmark_128x128x340_single_steps10400
uv run python scripts\generate_convergence_report.py
```

## 1D solver-alignment ladder

| Structure | RCWA ok | FDTD ok | max \|Rrcwa-Rtmm\| | max \|Rfdtd-Rtmm\| | max \|Tfdtd-Ttmm\| |
| --- | --- | --- | --- | --- | --- |
| lossless_slab | yes | yes | 1.57e-08 | 0.0002 | 0.004 |
| lossless_pixel_like_multilayer | yes | yes | 1.14e-06 | 0.0039 | 0.0026 |
| lossy_pixel_like_multilayer | yes | yes | 7.26e-07 | 0.0026 | 0.0011 |

### 1D slab: TMM vs torcwa RCWA vs 1D FDTD

![01 Slab Spectrum Alignment](/reports/convergence/alignment_slab/01_slab_spectrum_alignment.png)

*01 Slab Spectrum Alignment*

![01 Spectrum Alignment](/reports/convergence/alignment_slab/01_spectrum_alignment.png)

*01 Spectrum Alignment*

![02 Alignment Errors](/reports/convergence/alignment_slab/02_alignment_errors.png)

*02 Alignment Errors*

### 1D lossless multilayer: TMM vs torcwa RCWA vs 1D FDTD

![01 Spectrum Alignment](/reports/convergence/alignment_multilayer/01_spectrum_alignment.png)

*01 Spectrum Alignment*

![02 Alignment Errors](/reports/convergence/alignment_multilayer/02_alignment_errors.png)

*02 Alignment Errors*

![03 Fdtd Grid Convergence](/reports/convergence/alignment_multilayer/03_fdtd_grid_convergence.png)

*03 Fdtd Grid Convergence*

### 1D lossy pixel-like multilayer: TMM vs torcwa RCWA vs 1D FDTD

![01 Spectrum Alignment](/reports/convergence/alignment_lossy_multilayer/01_spectrum_alignment.png)

*01 Spectrum Alignment*

![02 Alignment Errors](/reports/convergence/alignment_lossy_multilayer/02_alignment_errors.png)

*02 Alignment Errors*

![03 Fdtd Grid Convergence](/reports/convergence/alignment_lossy_multilayer/03_fdtd_grid_convergence.png)

*03 Fdtd Grid Convergence*

## 2D FDTI/BDTI periodic trench

This benchmark uses one shared periodic trench geometry for both solvers. It is the first rung where FDTI and BDTI directionality matters.

| Mode | RCWA order | FDTD dx um | max \|dR\| | max \|dT\| | max \|dA\| | field leakage | aligned |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FDTI | 3 | 0.015 | 0.0266 | 0.027 | 0.0278 | 0.1305 | yes |
| BDTI | 3 | 0.015 | 0.0183 | 0.0088 | 0.019 | 0.096 | yes |

### 2D periodic FDTI/BDTI trench: torcwa RCWA vs 2D TE FDTD

![01 Shared Geometry](/reports/convergence/periodic_trench/01_shared_geometry.png)

*01 Shared Geometry*

![02 Spectrum Alignment](/reports/convergence/periodic_trench/02_spectrum_alignment.png)

*02 Spectrum Alignment*

![03 Rta Error](/reports/convergence/periodic_trench/03_rta_error.png)

*03 Rta Error*

![04 Fdtd Field Maps](/reports/convergence/periodic_trench/04_fdtd_field_maps.png)

*04 Fdtd Field Maps*

![05 Convergence](/reports/convergence/periodic_trench/05_convergence.png)

*05 Convergence*

## Full 2x2 Bayer pixel convergence

The pixel benchmark uses the real `PixelStack` path with FDTI/BDTI options, material-database complex indices, BARL layers, microlens slices, color filters, and photodiode integration windows. RCWA reports full-supercell R/T/A, while the scalar FDTD path reports localized-source collection and crosstalk proxies.

![Pixel convergence summary](/reports/convergence/pixel_convergence_summary.png)

*Pixel convergence summary generated from all available pixel benchmark metrics.*

| Grid | steps | sources | mode | dx um | c*time um | self frac | max neighbor | tail | warnings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 44x44x118 | 950 | all | FDTI | 0.0455 | 9.5962 | 0.7426 | 0.2003 | 0.278 | 1 |
| 44x44x118 | 950 | all | BDTI | 0.0455 | 9.5962 | 0.7426 | 0.2003 | 0.278 | 1 |
| 44x44x118 | 2200 | all | FDTI | 0.0455 | 22.2228 | 0.6041 | 0.2724 | 0.1077 | 0 |
| 44x44x118 | 2200 | all | BDTI | 0.0455 | 22.2228 | 0.6041 | 0.2728 | 0.1077 | 0 |
| 44x44x118 | 3500 | all | FDTI | 0.0455 | 35.3544 | 0.5831 | 0.2582 | 0.0471 | 0 |
| 44x44x118 | 3500 | all | BDTI | 0.0455 | 35.3544 | 0.5827 | 0.2609 | 0.047 | 0 |
| 64x64x170 | 3500 | all | FDTI | 0.0312 | 24.3794 | 0.5639 | 0.4434 | 0.3352 | 1 |
| 64x64x170 | 3500 | all | BDTI | 0.0312 | 24.3794 | 0.5677 | 0.4377 | 0.3353 | 1 |
| 64x64x170 | 5200 | all | FDTI | 0.0312 | 36.2208 | 0.5155 | 0.4367 | 0.2222 | 1 |
| 64x64x170 | 5200 | all | BDTI | 0.0312 | 36.2208 | 0.5192 | 0.4293 | 0.2223 | 1 |
| 128x128x340 | 10400 | single | FDTI | 0.0156 | 36.2208 | 0.0061 | 0.4945 | 0.22 | 1 |
| 128x128x340 | 10400 | single | BDTI | 0.0156 | 36.2208 | 0.0071 | 0.4933 | 0.2201 | 1 |

### Pixel plots

The images below are copied from the benchmark output folders. Use the geometry and field-slice plots to catch direction, indexing, and source-placement issues that scalar metrics alone can hide.

#### 2x2 pixel scalar FDTD, 44x44x118, 950 steps

![01 Geometry Slices](/reports/convergence/pixel_44x44x118_steps950/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_44x44x118_steps950/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_44x44x118_steps950/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_44x44x118_steps950/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

#### 2x2 pixel scalar FDTD, 44x44x118, 2200 steps

![01 Geometry Slices](/reports/convergence/pixel_44x44x118_steps2200/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_44x44x118_steps2200/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_44x44x118_steps2200/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_44x44x118_steps2200/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

#### 2x2 pixel scalar FDTD, 44x44x118, 3500 steps

![01 Geometry Slices](/reports/convergence/pixel_44x44x118_steps3500/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_44x44x118_steps3500/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_44x44x118_steps3500/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_44x44x118_steps3500/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

#### 2x2 pixel scalar FDTD, 64x64x170, 3500 steps

![01 Geometry Slices](/reports/convergence/pixel_64x64x170_steps3500/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_64x64x170_steps3500/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_64x64x170_steps3500/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_64x64x170_steps3500/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

#### 2x2 pixel scalar FDTD, 64x64x170, 5200 steps

![01 Geometry Slices](/reports/convergence/pixel_64x64x170_steps5200/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_64x64x170_steps5200/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_64x64x170_steps5200/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_64x64x170_steps5200/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

#### 2x2 pixel scalar FDTD, 128x128x340, single source, 10400 steps

![01 Geometry Slices](/reports/convergence/pixel_128x128x340_single_steps10400/01_geometry_slices.png)

*01 Geometry Slices*

![02 Rcwa Rta](/reports/convergence/pixel_128x128x340_single_steps10400/02_rcwa_rta.png)

*02 Rcwa Rta*

![03 Fdtd Crosstalk Matrix](/reports/convergence/pixel_128x128x340_single_steps10400/03_fdtd_crosstalk_matrix.png)

*03 Fdtd Crosstalk Matrix*

![04 Fdtd Field Slices](/reports/convergence/pixel_128x128x340_single_steps10400/04_fdtd_field_slices.png)

*04 Fdtd Field Slices*

## Visual smoke-test artifacts

These plots are not used as rigorous solver evidence. They are retained as fast visual tests for the FDTI/BDTI geometry, photodiode windows, and plotting pipeline.

### Visual CMOS pixel smoke test

![01 Geometry Overview](/reports/convergence/visual_cmos_pixel/01_geometry_overview.png)

*01 Geometry Overview*

![02 Rcwa Visual Test](/reports/convergence/visual_cmos_pixel/02_rcwa_visual_test.png)

*02 Rcwa Visual Test*

![03 Fdtd Visual Test](/reports/convergence/visual_cmos_pixel/03_fdtd_visual_test.png)

*03 Fdtd Visual Test*

![04 Rcwa Fdtd Comparison](/reports/convergence/visual_cmos_pixel/04_rcwa_fdtd_comparison.png)

*04 Rcwa Fdtd Comparison*

## Interpretation

- Use the 1D ladder to validate normalization, material loss, and monitor math before debugging full pixels.
- Use the periodic trench benchmark to compare FDTI and BDTI with the same geometry, boundary conditions, and R/T/A definitions.
- Treat full-pixel scalar FDTD as a visual convergence and crosstalk proxy until the energy tail falls below the selected threshold at the target grid.
- For final high-accuracy work, scale FDTD steps with grid refinement, run all four sources, and repeat the RCWA side with Fourier order and permittivity-grid sweeps.
