# Benchmarks & Validation

This document describes the methodology, reference data, and benchmark results used to verify the accuracy and performance of EM solvers in the COMPASS project.

---

## 1. Validation Methodology

To ensure the reliability of EM solvers, COMPASS employs a four-tier validation framework.

### 1.1 Validation Pyramid

```
        /\
       /  \        Level 4: Comparison with experimental data (published QE measurements)
      /    \
     /------\      Level 3: Cross-validation between solvers (torcwa vs grcwa vs meent vs FDTD)
    /        \
   /----------\    Level 2: Comparison with analytical reference solutions (Fresnel, TMM)
  /            \
 /--------------\  Level 1: Fundamental physics law verification (energy conservation, QE range)
```

Each validation level operates independently, and higher-level validations do not replace lower-level ones.

### 1.2 Validation Principles

| Principle | Description |
|-----------|-------------|
| Energy conservation | R + T + A = 1 (tolerance < 1%) |
| QE physical range | 0 <= QE <= 1 (all pixels, all wavelengths) |
| No NaN/Inf | No numerical anomalies in any output values |
| Convergence check | Results converge as resolution/order increases |
| Reproducibility | Identical results for identical inputs |

---

## 2. Analytical Reference Solutions

Solver outputs are validated against simple structures for which analytical solutions exist.

### 2.1 Fresnel Equations -- Single Interface Reflectance

An exact analytical solution exists for reflectance at a planar interface.

**Normal incidence Fresnel reflectance:**

```
R = |r|^2 = |(n1 - n2) / (n1 + n2)|^2
```

Reference values validated in the COMPASS benchmark test (`tests/benchmarks/test_fresnel_slab.py`):

| Interface | Wavelength (um) | n1 | n2 (complex) | Theoretical R | Tolerance |
|-----------|-----------------|-----|--------------|---------------|-----------|
| Air -> Glass | All | 1.000 | 1.500 + 0i | 0.0400 | 1e-6 |
| Air -> Si (550nm) | 0.550 | 1.000 | 4.08 + 0.028i | ~0.368 | 0.02 |
| Air -> SiO2 | 0.550 | 1.000 | 1.46 + 0i | ~0.035 | 1e-3 |
| Glass -> Air (reverse) | All | 1.500 | 1.000 | 0.0400 | 1e-6 |

**Angle-dependent validation:**

| Condition | Expected result | Test |
|-----------|-----------------|------|
| Brewster's angle (Air->Glass, 56.31 deg) | R_p = 0 | `test_brewster_angle` |
| Grazing incidence (89.9 deg) | R_s > 0.99, R_p > 0.98 | `test_grazing_incidence` |
| Reciprocity | R(n1->n2) = R(n2->n1) | `test_reciprocity` |

### 2.2 Transfer Matrix Method (TMM) -- Multilayer Thin Film Reference Solutions

For uniform multilayer structures, TMM (Transfer Matrix Method) yields results identical to RCWA at Fourier order = 0. These are used as reference solutions to validate the basic operation of the solvers.

**Single-layer anti-reflection coating validation:**

| Structure | Condition | Theoretical value | Notes |
|-----------|-----------|-------------------|-------|
| Air / SiO2 (93nm) / Si | 550nm, normal incidence | R ~ 0.12 | Single ARC |
| Air / Si3N4 (69nm) / Si | 550nm, normal incidence | R ~ 0.08 | High-index ARC |
| Air / MgF2 / SiO2 / Si | 550nm, normal incidence | R < 0.02 | Double-layer ARC |

These reference solutions are used to verify that the RCWA solvers operate correctly for uniform layers.

<ThinFilmReflectance />

### 2.3 Material Property Validation

`TestMaterialDBProperties` in `tests/benchmarks/test_fresnel_slab.py` validates the physical plausibility of built-in materials: Silicon (n~4.08 at 550nm, Green 2008), Air (n=1, k=0), SiO2 (n~1.46), Si3N4 (n~2.0), and Cauchy dispersion of Polymer (n increases at shorter wavelengths), among others.

---

## 3. Convergence Analysis

### 3.1 RCWA Fourier Order Convergence

The accuracy of RCWA depends on the Fourier order. Increasing the order improves accuracy, but computational cost grows as O(M^3) (M = (2N+1)^2 for 2D).

**Typical convergence behavior for a 1um pitch BSI pixel:**

| Fourier order | Number of harmonics (2D) | Matrix size | Relative QE change | Convergence status |
|---------------|--------------------------|-------------|---------------------|--------------------|
| 3 | 49 | 98x98 | Baseline | Not converged |
| 5 | 121 | 242x242 | ~5% | Not converged |
| 7 | 225 | 450x450 | ~2% | Near convergence |
| 9 | 361 | 722x722 | ~0.5% | Converged (recommended) |
| 11 | 529 | 1058x1058 | ~0.2% | Converged |
| 13 | 729 | 1458x1458 | < 0.1% | Well converged |
| 15 | 961 | 1922x1922 | < 0.05% | High precision |

**Convergence criterion**: Convergence is determined when the QE change between two consecutive orders is less than 1%.

<RCWAConvergenceDemo />

**Recommended orders by structure type:**

| Structure | Minimum order | Recommended order | Notes |
|-----------|---------------|-------------------|-------|
| Simple color filter pattern | 7 | 9 | Only CF boundaries present |
| With metal grid | 11 | 13--15 | High permittivity contrast |
| With DTI | 9 | 11--13 | SiO2/Si boundary |
| Fine nanostructures | 15 | 17--21 | Sub-wavelength features |

### 3.2 FDTD Grid Spacing Convergence

FDTD accuracy depends on the spatial grid spacing (dx, dy, dz). Generally, dx < lambda/(20*n_max) is recommended. For structures containing Si at 550nm, dx=20nm yields a QE change of ~1.5% (near convergence), while dx=10nm yields ~0.2% (well converged).

### 3.3 Microlens Staircase Approximation Convergence

In RCWA, microlenses are represented using a staircase approximation. Increasing the number of slices improves the accuracy of the lens volume representation.

Convergence behavior confirmed in the COMPASS benchmark (`tests/benchmarks/test_convergence.py`):

| Number of slices | Lens volume error | Convergence status |
|------------------|-------------------|--------------------|
| 5 | Baseline | Inaccurate |
| 10 | < 10% (vs 80) | Near convergence |
| 20 | < 5% (vs 80) | Converged |
| 40 | < 2% (vs 80) | Well converged |
| 80 | Baseline (highest resolution) | Reference value |

The volume difference between consecutive slice counts should decrease monotonically to indicate proper convergence. COMPASS uses 30 slices as the default.

---

## 4. Cross-Validation Between Solvers

### 4.1 RCWA Solver Comparison (torcwa vs grcwa vs meent)

The results of the three RCWA solvers are compared for the same pixel structure and the same Fourier order. Since all three solvers implement the same mathematical algorithm (RCWA), their results should agree within numerical error bounds.

**Acceptable discrepancy ranges:**

| Comparison metric | Acceptable range | Notes |
|-------------------|------------------|-------|
| QE absolute difference | < 0.005 (0.5%) | Same order, same factorization |
| QE relative error | < 1% | Relative to peak QE |
| Reflectance difference | < 0.002 | Across full wavelength range |
| Energy conservation deviation | < 0.01 | |R+T+A-1| |

**Typical comparison results (2x2 BSI pixel, 1um pitch, Fourier order 9):**

| Wavelength (nm) | torcwa QE(G) | grcwa QE(G) | meent QE(G) | Max difference |
|-----------------|-------------|-------------|-------------|----------------|
| 420 | ~0.15 | ~0.15 | ~0.15 | < 0.005 |
| 450 | ~0.32 | ~0.32 | ~0.32 | < 0.003 |
| 500 | ~0.52 | ~0.52 | ~0.52 | < 0.003 |
| 550 | ~0.58 | ~0.58 | ~0.58 | < 0.002 |
| 600 | ~0.45 | ~0.45 | ~0.45 | < 0.003 |
| 650 | ~0.28 | ~0.28 | ~0.28 | < 0.004 |
| 680 | ~0.18 | ~0.18 | ~0.18 | < 0.005 |

Note: The values above are representative and will vary depending on the actual structure and material parameters.

<SolverComparisonChart />

### 4.2 RCWA vs FDTD Comparison

Since RCWA and FDTD use different numerical methods, larger differences are expected compared to inter-RCWA solver comparisons. However, converged results from both methods should show good agreement.

**Acceptable discrepancy ranges:**

| Comparison metric | Acceptable range | Notes |
|-------------------|------------------|-------|
| QE absolute difference | < 0.02 (2%) | Based on converged results |
| QE relative error | < 5% | Relative to peak QE |
| Reflectance difference | < 0.01 | |
| Spectral shape agreement | Peak position within 10nm | |

The main sources of differences between RCWA and FDTD are microlens representation differences (staircase approximation vs subgrid approximation), dispersive media treatment methods, boundary conditions (periodic vs PML), and differences in convergence criteria.

---

## 5. Energy Conservation Validation

### 5.1 Fundamental Principle

By the law of energy conservation, the energy of incident light must be divided into reflection (R), transmission (T), and absorption (A):

```
R + T + A = 1
```

COMPASS validates energy conservation for all simulation results through the `EnergyBalance.check()` method in `compass/analysis/energy_balance.py`.

<EnergyBalanceDiagram />

### 5.2 Tolerance Criteria

| Condition | Tolerance | Action |
|-----------|-----------|--------|
| |R+T+A-1| < 0.01 (1%) | Normal | Accept result |
| 0.01 < |R+T+A-1| < 0.02 | Warning | Accept result + warning log |
| 0.02 < |R+T+A-1| < 0.05 | Retry | Automatic precision upgrade (AdaptivePrecisionRunner) |
| |R+T+A-1| > 0.05 | Failure | Reject result, manual review required |

### 5.3 Causes of Energy Conservation Violations and Remedies

| Violation type | Primary cause | Remedy |
|----------------|---------------|--------|
| R+T+A > 1 (energy creation) | Eigenvalue decomposition error, TF32 usage | Switch to float64, disable TF32 |
| R+T+A << 1 (energy loss) | T-matrix overflow, PML reflection | Use S-matrix, increase PML thickness |
| Violation at specific wavelengths only | Degenerate eigenvalues, high Q-factor | Eigenvalue broadening, precision upgrade |

`tests/benchmarks/test_energy_conservation.py` runs 11 tests covering perfect conservation, total reflection/total absorption, automatic absorption inference, tolerance boundaries, over/under unity, and single-wavelength violations.

---

## 6. Performance Benchmarks

### 6.1 Core Operation Performance

Performance criteria for core operations measured in `tests/benchmarks/test_performance.py`:

| Operation | Condition | Upper bound |
|-----------|-----------|-------------|
| `get_epsilon_spectrum` | Si, 41 wavelengths | 1.0 s |
| Full material spectrum | Built-in materials x 41 wavelengths | 2.0 s |
| PixelStack creation | 2x2 unit cell | 1.0 s |
| PixelStack creation | 4x4 unit cell | 2.0 s |
| `get_layer_slices` | nx=128 | 10 s |
| `get_permittivity_grid` | 128x128x128 | 60 s |

### 6.2 Solver Execution Times (Representative Values)

**Configuration**: 2x2 BSI pixel, 1um pitch, single wavelength (550nm), unpolarized

| Solver | Order/Resolution | CPU time | GPU time (NVIDIA A100) | Notes |
|--------|-----------------|----------|------------------------|-------|
| torcwa | Order 9 | ~5 s | ~0.3 s | Default solver |
| torcwa | Order 15 | ~40 s | ~2 s | High precision |
| grcwa | Order 9 | ~8 s | ~0.5 s | |
| meent | Order 9 | ~6 s | ~0.4 s | Analytical eigenvalue decomposition |
| flaport FDTD | dx=20nm | ~60 s | ~5 s | Time-domain |
| flaport FDTD | dx=10nm | ~300 s | ~25 s | High precision |

Note: The times above are representative estimates and may vary significantly depending on hardware and configuration.

FDTD is a time-domain method that can obtain the full spectrum in a single run. RCWA requires independent runs per wavelength but is easily parallelizable. For a wavelength sweep (14 wavelengths) on GPU, RCWA takes ~4--7s and FDTD takes ~5s.

### 6.3 Memory and Scaling

RCWA memory usage increases with Fourier order: ~0.3GB (f32) at order 9, ~1.2GB (f32) at order 15, and ~5GB (f32) at order 21 on GPU. Using float64 approximately doubles memory usage. FDTD memory varies greatly with grid spacing, using ~8GB on GPU at dx=10nm.

RCWA runtime scaling is O(M^3) with respect to matrix size M = (2N+1)^2, which translates to approximately O(N^6) with respect to order N. Compared to order 9, order 15 is approximately 11x slower and order 17 is approximately 20x slower.

---

## 7. Industry Validation Data

### 7.1 Published BSI Pixel QE Data

These are typical QE values for BSI CMOS image sensors reported in academic papers and industry presentations.

**Typical QE range for 1.0--1.4um pitch BSI pixels:**

| Color channel | Peak wavelength (nm) | QE range | Typical peak QE | Source |
|---------------|---------------------|----------|-----------------|--------|
| Blue (B) | 450--470 | 30--55% | ~45% | Multiple references |
| Green (G) | 530--560 | 45--70% | ~60% | Multiple references |
| Red (R) | 600--630 | 35--60% | ~50% | Multiple references |

**High-performance scientific BSI sensors (pitch > 3um):**

| Condition | Reported QE | Source |
|-----------|------------|--------|
| Visible peak (400--700nm) | > 75% | Industry datasheets |
| Broadband (260--400nm UV) | > 50% | Academic papers |
| Best performance (with nanostructures) | > 90% | Small, 2023 |

### 7.2 Comparison of COMPASS Simulation with Published Data

A qualitative comparison is made to determine whether simulation results are consistent with published data.

**Expected level of agreement:**

| Comparison metric | Expected agreement | Limiting factors |
|-------------------|--------------------|------------------|
| QE spectral shape | Excellent | Color filter model accuracy |
| Peak QE absolute value | Good (+-10%) | Material data accuracy, process variables |
| Crosstalk trends | Qualitative agreement | 3D structure simplification |
| Angular dependence | Good | Microlens profile accuracy |

The main sources of discrepancy between simulation and experiment are vendor-specific n,k differences in color filters, non-ideal geometries (tilted sidewalls, roughness), difficulty in reproducing illumination conditions, and process variables such as film thickness and alignment errors.

---

## 8. COMPASS Benchmark Suite

### 8.1 Test File Structure

```
tests/benchmarks/
  __init__.py
  test_fresnel_slab.py       # Fresnel equation and material property validation
  test_energy_conservation.py # Energy conservation validator tests
  test_convergence.py         # Geometry convergence tests
  test_performance.py         # Performance benchmarks (execution time)
```

### 8.2 Summary of Each Test File

| File | Number of classes | Number of tests | Key validation content |
|------|-------------------|-----------------|------------------------|
| `test_fresnel_slab.py` | 3 | 14 | Fresnel reflectance, material properties, epsilon relations |
| `test_energy_conservation.py` | 3 | 11 | Energy conservation valid/violation/edge cases |
| `test_convergence.py` | 3 | 9 | Microlens convergence, layer consistency, Bayer pattern |
| `test_performance.py` | 4 | 8 | MaterialDB, PixelStack, layer slices, permittivity grid performance |

### 8.3 Running Benchmarks

```bash
# Run all benchmarks
PYTHONPATH=. python3.11 -m pytest tests/benchmarks/ -v

# Run Fresnel validation only
PYTHONPATH=. python3.11 -m pytest tests/benchmarks/test_fresnel_slab.py -v

# Performance benchmarks (with timing output)
PYTHONPATH=. python3.11 -m pytest tests/benchmarks/test_performance.py -v -s

# Include slow tests
PYTHONPATH=. python3.11 -m pytest tests/benchmarks/ -v --run-slow
```

### 8.4 Running Solver Comparison Benchmarks

A detailed solver comparison workflow is documented in `docs/cookbook/solver-benchmark.md`.

```bash
# Run solver comparison script
python scripts/compare_solvers.py experiment=solver_comparison
```

This script runs torcwa/grcwa/meent on the same pixel structure and performs QE comparison, execution time comparison, energy conservation validation, and result visualization.

---

## 9. Validation Checklist

These are items that must be verified when adding a new solver or modifying an existing solver.

**Required validations (all changes):**
- [ ] All tests in `tests/benchmarks/` pass
- [ ] Energy conservation: |R+T+A-1| < 0.01 (all wavelengths)
- [ ] QE range: 0 <= QE <= 1 (all pixels)
- [ ] No NaN/Inf (all outputs)

**Recommended validations (major changes):**
- [ ] Cross-comparison with existing solvers (QE difference < 0.5%)
- [ ] Fourier order convergence check (order 7->15 sweep)
- [ ] GPU/CPU and float32/float64 result comparison

**Optional validations (performance changes):**
- [ ] No performance benchmark regression
- [ ] No memory usage regression
