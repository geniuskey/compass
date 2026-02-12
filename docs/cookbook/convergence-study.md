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

### When do you need high orders?

The required Fourier order depends on the smallest feature in your pixel structure relative to the simulation domain:

- **Metal grid**: 0.05 um features in a 2 um domain requires Nyquist sampling of at least ~40 harmonics per axis
- **Color filter only**: Smooth layers converge quickly with low orders
- **Microlens + metal grid**: The metal grid drives the convergence requirement

::: warning
Under-resolved metal grid features cause artificial scattering and incorrect absorption. Always verify convergence when metal grids are present.
:::

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
PYTHONPATH=. python3.11 scripts/convergence_study.py --sweep full_spectrum
```

## Recommended Parameters

| Parameter | Fast | Default | Converged |
|-----------|------|---------|-----------|
| grcwa fourier_order | [25, 25] | [9, 9] | [121, 121] |
| torcwa fourier_order | [5, 5] | [9, 9] | [9, 9] |
| n_lens_slices | 15 | 30 | 50 |
| grid_multiplier | 3 | 3 | 3 |

::: warning grcwa numerical stability
grcwa exhibits numerical instability (R → huge values) at certain `nG` values (e.g., 81, 169, 225). The recommended converged value `nG=121` is verified stable. If you observe R > 1 or negative absorption, try adjusting nG by ±10.
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

## Key Takeaways

1. **Fourier order is the most critical parameter** -- it has the largest impact on accuracy and runtime
2. **Metal grid features drive convergence requirements** -- smooth structures converge at much lower orders
3. **grcwa converges faster** (nG=49 at 0.95s) **than torcwa** ([9,9] at 55s) for equivalent accuracy
4. **n_lens_slices = 15-20** is sufficient for most microlens shapes
5. **grid_multiplier = 3** is sufficient for typical pixel geometries
6. **grcwa has numerical instability** at certain nG values -- always verify R + T + A ≈ 1
