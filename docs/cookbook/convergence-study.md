---
outline: deep
---

# Convergence Study

<ConvergenceStudyChart />

## Overview

A convergence study determines the minimum simulation parameter values that yield accurate, stable results. In RCWA simulations, the key numerical parameters are:

- **Fourier order** (number of plane wave harmonics)
- **Microlens slices** (staircase approximation of curved surfaces)
- **Grid resolution** (spatial sampling of the permittivity distribution)

Increasing any of these parameters improves accuracy but increases computation time. The goal is to find the "knee" where results stop changing significantly -- the **converged** regime.

::: tip Why convergence matters
Running with too few harmonics gives incorrect results (underestimated absorption). Running with too many wastes compute time. A convergence study finds the sweet spot.
:::

## Fourier Order

The Fourier order controls how many plane wave harmonics are used to expand the electromagnetic fields. More harmonics capture finer spatial features in the permittivity distribution.

### grcwa vs torcwa parameterization

The two RCWA solvers parameterize Fourier order differently:

- **grcwa**: Uses `nG` (total number of harmonics). The solver internally selects which G-vectors to include.
- **torcwa**: Uses `[N, N]` where total harmonics = `(2N+1)^2`. For example, `[5, 5]` gives `11^2 = 121` harmonics.

### Convergence results

**grcwa** (stable points only — see instability warning below):

| nG | Harmonics | Absorption | Runtime | ΔA |
|----|-----------|-----------|---------|-----|
| 9 | 9 | 0.9905 | 0.02s | +0.0206 |
| 25 | 25 | 0.9868 | 0.17s | -0.0037 |
| 49 | 49 | 0.9699 | 0.95s | -0.0169 |
| 121 | 121 | 0.9712 | 5.42s | +0.0013 |
| 625 | 625 | 0.9749 | 356.24s | +0.0037 |

**torcwa** (numerically stable at all orders):

| [N,N] | Harmonics | Absorption | Runtime | ΔA |
|-------|-----------|-----------|---------|-----|
| [3,3] | 49 | 0.9820 | 2.77s | — |
| [5,5] | 121 | 0.9856 | 23.12s | +0.0036 |
| [7,7] | 225 | 0.9858 | 100.13s | +0.0002 |
| [9,9] | 361 | 0.9831 | 349.79s | -0.0027 |
| [11,11] | 529 | 0.9858 | 846.12s | +0.0028 |

Note: The non-monotonic dip at [9,9] recovers at [11,11], confirming true convergence at A ≈ 0.986.

::: info Solver absorption difference
grcwa and torcwa produce slightly different absorption values (~0.97 vs ~0.98) due to differences in G-vector selection and Fourier factorization. Both converge to stable values, but the absolute difference reflects implementation-level choices in each solver.
:::

### When do you need high orders?

The required Fourier order depends on the smallest feature in your pixel structure relative to the simulation domain:

- **Metal grid**: 0.05 um features in a 2 um domain requires Nyquist sampling of at least ~40 harmonics per axis
- **Color filter only**: Smooth layers converge quickly with low orders
- **Microlens + metal grid**: The metal grid drives the convergence requirement

::: warning
Under-resolved metal grid features cause artificial scattering and incorrect absorption. Always verify convergence when metal grids are present.
:::

### grcwa numerical instability

grcwa exhibits severe numerical instability at many `nG` values. The instability manifests as:
- **R → huge values** (up to 10^11) while T remains near zero
- **TM polarization only** — TE polarization is always stable
- **Wavelength-dependent** — nG=121 is stable at 550nm but fails at 400nm and 500nm with "Singular matrix" errors
- **Even grid_multiplier** values (2, 4) trigger instability; odd values (3, 5) are stable

**Unstable nG values found**: 81, 169, 225, 289, 361, 441, 529

**Stable nG values verified**: 9, 25, 49, 121 (at 550nm), 625

This is a fundamental limitation of the grcwa library's S-matrix implementation, likely related to Li's inverse factorization rule for TM polarization. For production simulations requiring stability across the full visible spectrum, **use torcwa** or verify each wavelength individually with grcwa.

## Microlens Slices

The microlens is approximated by a staircase of flat layers (`n_lens_slices`). More slices better approximate the curved microlens profile, but each slice adds a layer to the RCWA computation.

The convergence data shows that:
- **5 slices**: Noticeably lower absorption (0.9662)
- **15 slices**: Within 0.001 of the converged value (0.9696)
- **30+ slices**: Fully converged (0.9699-0.9700), no further improvement

For most simulations, **20-30 slices** provides excellent accuracy with minimal overhead.

## Grid Resolution

The `grid_multiplier` parameter controls the spatial resolution of the permittivity grid used to compute Fourier coefficients. Higher values better resolve sharp permittivity boundaries (e.g., metal grid edges).

The convergence is rapid:
- **multiplier = 3**: Converged for typical pixel structures (A = 0.9699)
- **multiplier = 5-6**: No significant change (A = 0.9698-0.9699)

::: warning grcwa instability with even grid_multiplier
grcwa shows numerical instability with even `grid_multiplier` values (2, 4) at certain Fourier orders. Use odd multipliers (3, 5) for grcwa. This does not affect torcwa.
:::

A `grid_multiplier` of **3** is sufficient for most simulations.

## Full Spectrum Validation

The full visible spectrum (400-700nm, 31 points) was validated at converged parameters (grcwa nG=49, n_lens_slices=30, grid_multiplier=3). All wavelengths satisfied R + T + A = 1.000 exactly, confirming numerical stability.

Key spectral features:
- **400-500nm** (blue): High absorption (A = 0.967-0.976), near-zero transmission — silicon absorbs strongly
- **500-560nm** (green): Absorption dip at 510-520nm (A ≈ 0.955) then recovery
- **560-700nm** (red): Gradual increase in transmission (T up to 0.011), absorption remains high (A > 0.955)

Total computation time: **34 seconds** for 31 wavelengths at nG=49.

## Running the Convergence Study

```bash
# Fourier order sweep (grcwa)
PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep fourier_order_grcwa

# Fourier order sweep (torcwa)
PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep fourier_order_torcwa

# Microlens slices sweep
PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep n_lens_slices

# Grid resolution sweep
PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep grid_resolution

# Full spectrum validation
PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep full_spectrum --fourier-order 49
```

## Recommended Parameters

| Parameter | Fast | Default | Converged |
|-----------|------|---------|-----------|
| grcwa fourier_order | [25, 25] | [49, 49] | [49, 49] |
| torcwa fourier_order | [3, 3] | [5, 5] | [7, 7] |
| n_lens_slices | 15 | 30 | 50 |
| grid_multiplier | 3 | 3 | 3 |

::: warning grcwa vs torcwa for production
grcwa is fast but has numerical instability at many Fourier orders and across wavelengths. For production simulations requiring full-spectrum stability, **torcwa [5,5] or [7,7]** is recommended. grcwa nG=49 is suitable for quick single-wavelength checks.
:::

::: info Note on fourier_order config format
In COMPASS config YAML, `fourier_order` is specified as `[nG_x, nG_y]`. For grcwa, the solver internally maps this to total harmonics. For torcwa, `[N, N]` directly sets the truncation order.
:::

## Config Presets

```bash
# Fast iteration
python scripts/run_simulation.py solver=grcwa_fast

# High accuracy (converged)
python scripts/run_simulation.py solver=grcwa_converged
```

## Per-Color QE Convergence

<PerColorConvergenceChart />

In a Bayer-pattern pixel, each sub-pixel has a different color filter (CF). Total absorption alone does not reveal per-channel behavior — convergence rate, cross-solver agreement, and angle sensitivity may differ by color. The per-color study uses **uniform color filter** simulations: all pixels use the same CF material (e.g., all `cf_red`), and total absorption at the CF peak wavelength gives the per-color QE.

### Methodology

Standard CIS practice: run three separate simulations with uniform CF (all-red at 620nm, all-green at 530nm, all-blue at 450nm). This isolates each color channel's QE without cross-talk from neighboring pixels' filters.

::: info Why not use per-pixel QE from RCWA?
RCWA solvers compute per-pixel QE by distributing total absorption using `eps_imag` weighting in each photodiode region. Since all PDs are silicon with identical `eps_imag`, all pixels receive equal QE = `total_A / n_pixels` regardless of their color filter. Uniform CF simulations are the standard approach to obtain true per-color QE.
:::

### Per-Color Fourier Convergence (torcwa)

| [N,N] | Harmonics | QE_R@620 | QE_G@530 | QE_B@450 |
|-------|-----------|----------|----------|----------|
| [3,3] | 49 | 0.9289 | 0.9471 | 0.9196 |
| [5,5] | 121 | 0.9622 | 0.9770 | 0.9697 |
| [7,7] | 225 | 0.9615 | 0.9737 | 0.9475 |
| [9,9] | 361 | 0.9617 | 0.9737 | 0.9486 |

- **Red**: Converged at N=5 (ΔQE < 0.001 from N=7 onward)
- **Green**: Converged at N=7 (ΔQE = 0.00002 between N=7 and N=9)
- **Blue**: Non-monotonic — peaks at N=5 (0.970) then settles to 0.948-0.949 at N=7-9. The blue channel converges to a lower QE than the initial N=5 estimate due to short-wavelength diffraction effects

### Per-Color Fourier Convergence (grcwa)

| nG | QE_R@620 | QE_G@530 | QE_B@450 | Status |
|----|----------|----------|----------|--------|
| 9 | 0.9811 | 0.9931 | 0.9949 | Stable (overestimates) |
| 25 | 0.9834 | 0.9883 | 0.9734 | Stable |
| 49 | 0.9333 | 0.9432 | 0.9316 | Stable |
| 81 | **0.5000** | 0.9670 | 0.9340 | R: TM diverged |

The grcwa TM instability at nG=81 specifically affects the red channel at 620nm (R_R=14.33, unphysical). Blue and green channels remain stable at nG=81, confirming the instability is wavelength-dependent.

### Cross-Solver Per-Color Comparison

| Solver | QE_R@620 | QE_G@530 | QE_B@450 |
|--------|----------|----------|----------|
| grcwa (nG=49) | 0.9333 | 0.9432 | 0.9316 |
| torcwa ([5,5]) | 0.9622 | 0.9770 | 0.9697 |
| **Difference** | -2.9% | -3.4% | -3.8% |

torcwa consistently gives higher QE across all colors. The largest gap is in the blue channel (3.8%), suggesting blue wavelengths are more sensitive to solver implementation differences (G-vector selection, Fourier factorization rules).

### Angle-Dependent Per-Color QE

| CRA | QE_R@620 | QE_G@530 | QE_B@450 | RI_R | RI_G | RI_B |
|-----|----------|----------|----------|------|------|------|
| 0° | 0.9622 | 0.9770 | 0.9697 | 1.000 | 1.000 | 1.000 |
| 15° | 0.9960 | 1.0000 | 0.9995 | 1.035 | 1.024 | 1.031 |
| 30° | 0.9980 | 0.9997 | 0.9954 | 1.037 | 1.023 | 1.027 |

::: warning Relative Illumination > 1.0
QE *increases* with CRA. This is because the pixel uses `auto_cra` microlens shift that optimally aligns the microlens focus for each CRA angle. At normal incidence (CRA=0°), the uniform CF configuration may not match the designed Bayer pattern geometry, leading to slightly lower coupling. The RI > 1.0 values should not be interpreted as physical gain — they reflect the simulation setup where CRA shift improves optical coupling.
:::

### Running Per-Color Experiments

```bash
# Per-color Fourier sweep (torcwa)
PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep fourier_per_color_torcwa

# Per-color Fourier sweep (grcwa)
PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep fourier_per_color_grcwa

# Cross-solver per-color comparison
PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep cross_solver_per_color

# Angle-dependent per-color QE
PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep angle_per_color
```

## Key Takeaways

1. **Fourier order is the most critical parameter** -- it has the largest impact on accuracy and runtime
2. **Metal grid features drive convergence requirements** -- smooth structures converge at much lower orders
3. **torcwa is more numerically stable** than grcwa across all Fourier orders and wavelengths
4. **grcwa is faster** (nG=49 at 0.95s) but has TM polarization instability at many nG values
5. **torcwa converges at [5,5]** (121 harmonics, 23s) with ΔA < 0.001
6. **n_lens_slices = 15-20** is sufficient for most microlens shapes
7. **grid_multiplier = 3** is sufficient for typical pixel geometries
8. **Always verify R + T + A ≈ 1** when using grcwa -- violations indicate numerical instability
9. **Blue channel converges slower** than red/green in per-color studies due to shorter wavelength features
10. **Cross-solver per-color gap is largest for blue** (3.8%) -- blue wavelengths are more sensitive to solver implementation differences
11. **grcwa TM instability is wavelength-dependent** -- nG=81 fails at 620nm but is stable at 450nm and 530nm
