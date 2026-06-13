---
outline: deep
---

# Performance Benchmark

_Generated on 2026-06-11 from local CPU/GPU timing runs._

This report publishes a lightweight performance baseline for the code paths that developers exercise most often: material lookup, PixelStack geometry generation, TMM sweeps, and low-order torcwa RCWA sweeps.

## Executive summary

- All-material 41-wavelength lookup median: 19.7626 ms.
- PixelStack construction median: 102.5448 ms for 2x2, 105.3031 ms for 4x4.
- TMM 31-wavelength sweep median: 0.019 s; fitted cost 0.5786 ms/wavelength.
- torcwa CPU 5-wavelength low-order sweep median: 2.8404 s; fitted cost 604.8363 ms/wavelength.
- CUDA was not available in this environment, so GPU rows are recorded as not available.

::: warning Scope
These are local timing numbers, not universal performance guarantees. The RCWA case is a low-order smoke benchmark, and traced memory excludes some native library allocations such as PyTorch/CUDA allocator internals.
:::

## Environment

| Field | Value |
| --- | --- |
| Platform | Windows-11-10.0.26200-SP0 |
| Python | 3.12.12 |
| Machine | AMD64 |
| Processor | Intel64 Family 6 Model 151 Stepping 5, GenuineIntel |
| CPU count | 12 |
| Torch | 2.11.0+cpu |
| CUDA available | no |
| CUDA devices | 0 |
| CUDA device name | n/a |
| torcwa available | yes |
| torcwa version | 0.1.4.2 |

## Core CPU Operations

![Core operation runtime](/reports/performance-benchmark/01_core_operation_runtime.png)

| Category | Benchmark | Size | Repeats | Median ms | Min ms | Max ms | Peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| material | Silicon epsilon spectrum lookup | silicon x 41 wl | 5 | 0.5114 | 0.3746 | 0.6381 | 0.0022 |
| material | All-material spectrum lookup | 43 materials x 41 wl | 5 | 19.7626 | 18.7642 | 25.9281 | 0.0036 |
| construction | PixelStack construction | 2x2 Bayer | 5 | 102.5448 | 88.6468 | 119.9351 | 0.188 |
| construction | PixelStack construction | 4x4 TetraCell | 5 | 105.3031 | 99.3698 | 119.5264 | 0.1883 |
| layer_slices | get_layer_slices | 32x32, 8 lens slices | 5 | 4.2025 | 2.7281 | 5.0091 | 0.4285 |
| layer_slices | get_layer_slices | 64x64, 8 lens slices | 5 | 4.646 | 3.3389 | 6.5738 | 1.6473 |
| layer_slices | get_layer_slices | 128x128, 8 lens slices | 5 | 6.3995 | 5.3761 | 14.7934 | 6.5223 |
| layer_slices | get_layer_slices | 192x192, 8 lens slices | 5 | 14.9338 | 13.8587 | 16.3564 | 14.6473 |
| permittivity_grid | get_permittivity_grid | 32x32x64 | 4 | 8.2513 | 7.3968 | 9.746 | 1.7842 |
| permittivity_grid | get_permittivity_grid | 64x64x64 | 4 | 9.9216 | 9.7119 | 10.8755 | 7.0342 |
| permittivity_grid | get_permittivity_grid | 96x96x64 | 4 | 15.8072 | 13.7619 | 16.9709 | 15.7842 |
| permittivity_grid | get_permittivity_grid | 128x128x64 | 4 | 27.1593 | 23.096 | 29.8962 | 28.0342 |

## Geometry Scaling

![Geometry scaling](/reports/performance-benchmark/02_geometry_scaling.png)

![Memory profile](/reports/performance-benchmark/03_memory_profile.png)

## Solver Sweep Cost

![Solver wavelength scaling](/reports/performance-benchmark/04_solver_wavelength_scaling.png)

| Solver | Device | Wavelengths | Status | Repeats | Median s | ms / wl | wl / s | max energy residual | traced MB | GPU MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tmm | cpu | 1 | ok | 3 | 0.0011 | 1.0969 | 911.6603 | 0 | 0.0107 | n/a |
| tmm | cpu | 11 | ok | 3 | 0.0077 | 0.6998 | 1428.9797 | 1.11e-16 | 0.0118 | n/a |
| tmm | cpu | 31 | ok | 3 | 0.019 | 0.6137 | 1629.3664 | 1.11e-16 | 0.0131 | n/a |
| tmm | cpu | 61 | ok | 3 | 0.0321 | 0.5268 | 1898.2891 | 1.11e-16 | 0.0152 | n/a |
| tmm | cpu | 101 | ok | 3 | 0.0607 | 0.6008 | 1664.4282 | 1.11e-16 | 0.0196 | n/a |
| torcwa | cpu | 1 | ok | 3 | 0.6147 | 614.7073 | 1.6268 | 0 | 1.538 | n/a |
| torcwa | cpu | 3 | ok | 3 | 1.7693 | 589.7522 | 1.6956 | 0 | 2.8789 | n/a |
| torcwa | cpu | 5 | ok | 3 | 2.8404 | 568.0826 | 1.7603 | 0 | 2.8898 | n/a |
| torcwa | cpu | 9 | ok | 3 | 5.4565 | 606.2821 | 1.6494 | 0 | 2.9098 | n/a |
| torcwa | cuda | n/a | not_available | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

## Interpretation

- TMM remains the fast path for stack screening and wavelength-dense BARL sweeps.
- Low-order torcwa CPU runtime scales close to linearly with wavelength count, so larger reports should budget by wavelength first, then Fourier order.
- Geometry generation is cheap enough for repeated report generation at the tested resolutions, but 3D permittivity grids dominate traced memory.
- The GPU row is intentionally conditional; machines without CUDA still generate a complete CPU report.

## Regeneration

```powershell
uv run python scripts\generate_performance_benchmark_report.py
```

Generated metrics are stored at `docs/public/reports/performance-benchmark/performance_benchmark_metrics.json`.
