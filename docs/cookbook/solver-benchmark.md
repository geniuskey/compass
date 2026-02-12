# Solver Comparison & Selection Guide

<SolverComparisonChart />

## Overview

COMPASS supports multiple electromagnetic solver families for pixel simulation: **TMM** (Transfer Matrix Method) for fast 1D planar analysis, **RCWA** (Rigorous Coupled-Wave Analysis) with four implementations (torcwa, grcwa, meent, fmmax) for 2D periodic structures, and **FDTD** (Finite-Difference Time-Domain) solvers for arbitrary geometries. This guide consolidates benchmark results, cross-solver validation data, and practical selection guidance to help you choose the right solver for your task.

## Part 1: RCWA Solver Benchmark

This section compares RCWA solvers (torcwa, grcwa, meent) on the same pixel structure, measuring accuracy and performance.

### Goal

Run the same BSI pixel simulation with all available RCWA solvers, then compare QE results and execution times.

### Setup

```python
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from compass.runners.single_run import SingleRunner
from compass.analysis.solver_comparison import SolverComparison
from compass.visualization.qe_plot import plot_qe_comparison

pixel_config = {
    "pitch": 1.0,
    "unit_cell": [2, 2],
    "bayer_map": [["R", "G"], ["G", "B"]],
    "layers": {
        "air": {"thickness": 1.0, "material": "air"},
        "microlens": {
            "enabled": True, "height": 0.6,
            "radius_x": 0.48, "radius_y": 0.48,
            "material": "polymer_n1p56",
            "profile": {"type": "superellipse", "n": 2.5, "alpha": 1.0},
            "shift": {"mode": "none"},
        },
        "planarization": {"thickness": 0.3, "material": "sio2"},
        "color_filter": {
            "thickness": 0.6,
            "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
            "grid": {"enabled": True, "width": 0.05, "material": "tungsten"},
        },
        "barl": {"layers": [
            {"thickness": 0.010, "material": "sio2"},
            {"thickness": 0.025, "material": "hfo2"},
            {"thickness": 0.015, "material": "sio2"},
            {"thickness": 0.030, "material": "si3n4"},
        ]},
        "silicon": {
            "thickness": 3.0, "material": "silicon",
            "photodiode": {"position": [0, 0, 0.5], "size": [0.7, 0.7, 2.0]},
            "dti": {"enabled": True, "width": 0.1, "material": "sio2"},
        },
    },
}

source_config = {
    "wavelength": {"mode": "sweep", "sweep": {"start": 0.42, "stop": 0.68, "step": 0.02}},
    "polarization": "unpolarized",
}
```

### Run all RCWA solvers

```python
solvers = [
    {"name": "torcwa", "type": "rcwa", "params": {"fourier_order": [9, 9]}},
    {"name": "grcwa",  "type": "rcwa", "params": {"fourier_order": [9, 9]}},
    {"name": "meent",  "type": "rcwa", "params": {"fourier_order": [9, 9]}},
]

results = []
labels = []

for solver_cfg in solvers:
    config = {
        "pixel": pixel_config,
        "solver": {
            **solver_cfg,
            "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
        },
        "source": source_config,
        "compute": {"backend": "auto"},
    }

    try:
        result = SingleRunner.run(config)
        results.append(result)
        labels.append(solver_cfg["name"])
        runtime = result.metadata.get("runtime_seconds", 0)
        print(f"{solver_cfg['name']}: {runtime:.2f}s")
    except Exception as e:
        print(f"{solver_cfg['name']}: FAILED - {e}")
```

### Compare QE spectra

```python
if len(results) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_qe_comparison(results, labels, ax=ax)
    ax.set_title("RCWA Solver Comparison")
    plt.tight_layout()
    plt.savefig("solver_comparison_qe.png", dpi=150)
    plt.show()
```

### Quantitative comparison

```python
if len(results) >= 2:
    comparison = SolverComparison(results, labels)
    summary = comparison.summary()

    print("\n=== Comparison Summary ===")
    print("\nMax QE difference (absolute):")
    for key, val in summary["max_qe_diff"].items():
        print(f"  {key}: {val:.5f}")

    print("\nMax QE relative error (%):")
    for key, val in summary["max_qe_relative_error_pct"].items():
        print(f"  {key}: {val:.2f}%")

    print("\nRuntimes:")
    for solver, runtime in summary["runtimes_seconds"].items():
        print(f"  {solver}: {runtime:.2f}s")
```

<RCWAConvergenceDemo />

<FourierOrderDemo />

### Fourier order convergence comparison

Compare how each solver converges with Fourier order:

```python
orders = [5, 7, 9, 11, 13, 15]
convergence_data = {s["name"]: [] for s in solvers}

for order in orders:
    for solver_cfg in solvers:
        config = {
            "pixel": pixel_config,
            "solver": {
                **solver_cfg,
                "params": {"fourier_order": [order, order]},
                "stability": {"precision_strategy": "mixed"},
            },
            "source": {
                "wavelength": {"mode": "single", "value": 0.55},
                "polarization": "unpolarized",
            },
            "compute": {"backend": "auto"},
        }

        try:
            result = SingleRunner.run(config)
            avg_qe = np.mean([qe[0] for qe in result.qe_per_pixel.values()])
            convergence_data[solver_cfg["name"]].append(avg_qe)
        except:
            convergence_data[solver_cfg["name"]].append(np.nan)

    print(f"Order {order}: done")

# Plot convergence
plt.figure(figsize=(8, 5))
for name, qe_values in convergence_data.items():
    plt.plot(orders, qe_values, "o-", label=name, linewidth=2)

plt.xlabel("Fourier Order")
plt.ylabel("Average QE at 550 nm")
plt.title("Convergence: QE vs Fourier Order")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("convergence_comparison.png", dpi=150)
```

### Runtime scaling

Measure how runtime scales with Fourier order:

```python
orders = [5, 7, 9, 11, 13, 15, 17]
timing_data = {s["name"]: [] for s in solvers}

for order in orders:
    for solver_cfg in solvers:
        config = {
            "pixel": pixel_config,
            "solver": {
                **solver_cfg,
                "params": {"fourier_order": [order, order]},
                "stability": {"precision_strategy": "mixed"},
            },
            "source": {
                "wavelength": {"mode": "single", "value": 0.55},
                "polarization": "TE",  # Single pol for timing
            },
            "compute": {"backend": "auto"},
        }

        try:
            result = SingleRunner.run(config)
            t = result.metadata.get("runtime_seconds", 0)
            timing_data[solver_cfg["name"]].append(t)
        except:
            timing_data[solver_cfg["name"]].append(np.nan)

# Plot
plt.figure(figsize=(8, 5))
for name, times in timing_data.items():
    plt.plot(orders, times, "o-", label=name, linewidth=2)

plt.xlabel("Fourier Order")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime Scaling vs Fourier Order")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale("log")
plt.tight_layout()
plt.savefig("runtime_scaling.png", dpi=150)
```

### Expected results

- **QE agreement**: All RCWA solvers should agree within 0.5-1% QE at the same Fourier order.
- **Convergence**: All solvers converge to the same QE as order increases. Convergence rate may differ slightly.
- **Runtime**: torcwa is typically fastest on GPU. grcwa and meent are competitive.
- **FDTD vs RCWA**: Should agree within 1-2% QE. FDTD is significantly slower for single-wavelength runs.

## Part 2: TMM vs RCWA

Cross-validation between TMM (1D) and RCWA (2D/3D) solvers confirms that different numerical methods produce physically consistent results and helps identify where 1D approximations break down relative to full 2D electromagnetic solutions.

### Overview

Three solvers are compared on the same pixel structure:

- **TMM** (Transfer Matrix Method): 1D analytical solver that treats each layer as an infinite uniform slab. Ignores lateral patterning (microlens shape, Bayer CF pattern, DTI). Extremely fast (~3 ms for a full sweep).
- **torcwa**: PyTorch-based 2D RCWA solver using the S-matrix algorithm. Handles full lateral structure including microlens profile, color filter Bayer pattern, and DTI boundaries.
- **grcwa**: NumPy-based 2D RCWA solver with autograd support. Independent RCWA implementation used to cross-check torcwa results.

::: info Simulation Parameters
- **Pixel**: 1.0 um pitch BSI, 2x2 RGGB Bayer
- **Stack**: 9 layers (air / microlens / planarization / CF / 4-layer BARL / silicon)
- **RCWA**: Fourier order [3,3], 5 microlens slices, 64x64 grid
- **Source**: Normal incidence, unpolarized, 380-780 nm (20 nm step)
- **Platform**: macOS (Apple Silicon), CPU only
:::

### Interactive Comparison

<CrossSolverValidation />

### Wavelength Sweep Results

#### Full Data Table

Complete R, T, A values for all 21 wavelengths across the three solvers. Light propagates from air (top) through the stack into silicon (bottom).

| λ (nm) | TMM R | TMM T | TMM A | torcwa R | torcwa T | torcwa A | grcwa R | grcwa T | grcwa A |
|-------:|------:|------:|------:|---------:|---------:|---------:|--------:|--------:|--------:|
| 380 | 0.0401 | 0.0517 | 0.9081 | 0.0147 | 0.0000 | 0.9853 | 0.0195 | 0.0000 | 0.9805 |
| 400 | 0.0529 | 0.0722 | 0.8750 | 0.0239 | 0.0000 | 0.9761 | 0.0139 | 0.0000 | 0.9861 |
| 420 | 0.0961 | 0.0830 | 0.8210 | 0.0021 | 0.0000 | 0.9979 | 0.0131 | 0.0000 | 0.9869 |
| 440 | 0.0348 | 0.1085 | 0.8566 | 0.0069 | 0.0000 | 0.9931 | 0.0125 | 0.0000 | 0.9875 |
| 460 | 0.0633 | 0.1443 | 0.7923 | 0.0018 | 0.0000 | 0.9982 | 0.0102 | 0.0000 | 0.9898 |
| 480 | 0.0324 | 0.2549 | 0.7127 | 0.0138 | 0.0000 | 0.9862 | 0.0119 | 0.0000 | 0.9881 |
| 500 | 0.1461 | 0.4549 | 0.3990 | 0.0186 | 0.0000 | 0.9814 | 0.0148 | 0.0000 | 0.9852 |
| 520 | 0.0088 | 0.9135 | 0.0777 | 0.0258 | 0.0000 | 0.9742 | 0.0144 | 0.0000 | 0.9856 |
| 540 | 0.2405 | 0.7023 | 0.0572 | 0.0131 | 0.0001 | 0.9868 | 0.0121 | 0.0000 | 0.9879 |
| 560 | 0.1697 | 0.4775 | 0.3528 | 0.0013 | 0.0003 | 0.9984 | 0.0114 | 0.0000 | 0.9886 |
| 580 | 0.0280 | 0.3331 | 0.6389 | 0.0021 | 0.0008 | 0.9971 | 0.0139 | 0.0000 | 0.9861 |
| 600 | 0.0387 | 0.2423 | 0.7190 | 0.0077 | 0.0026 | 0.9897 | 0.0176 | 0.0000 | 0.9824 |
| 620 | 0.0477 | 0.2175 | 0.7349 | 0.0137 | 0.0061 | 0.9802 | 0.0202 | 0.0000 | 0.9797 |
| 640 | 0.0180 | 0.2258 | 0.7563 | 0.0223 | 0.0093 | 0.9684 | 0.0206 | 0.0001 | 0.9793 |
| 660 | 0.0209 | 0.2336 | 0.7455 | 0.0126 | 0.0113 | 0.9761 | 0.0193 | 0.0001 | 0.9807 |
| 680 | 0.0781 | 0.2284 | 0.6935 | 0.0116 | 0.0126 | 0.9757 | 0.0173 | 0.0001 | 0.9826 |
| 700 | 0.1199 | 0.2252 | 0.6549 | 0.0093 | 0.0130 | 0.9777 | 0.0159 | 0.0001 | 0.9840 |
| 720 | 0.1091 | 0.2345 | 0.6564 | 0.0055 | 0.0140 | 0.9805 | 0.0161 | 0.0002 | 0.9837 |
| 740 | 0.0664 | 0.2522 | 0.6815 | 0.0032 | 0.0147 | 0.9821 | 0.0178 | 0.0003 | 0.9819 |
| 760 | 0.0372 | 0.2667 | 0.6961 | 0.0045 | 0.0142 | 0.9813 | 0.0205 | 0.0004 | 0.9792 |
| 780 | 0.0430 | 0.2721 | 0.6849 | 0.0116 | 0.0145 | 0.9739 | 0.0231 | 0.0004 | 0.9765 |

### Key Observations

#### 1D vs 2D Differences

The TMM and RCWA results differ significantly, and understanding why is critical for choosing the right solver for a given task.

##### 1. TMM vs RCWA Absorption

TMM shows absorption ranging from 5% to 91%, while RCWA consistently shows 97-100% absorption across the entire visible spectrum. The 3 um silicon layer absorbs nearly all light that successfully enters it. In TMM (1D), thin-film interference between planar layers creates constructive and destructive patterns that open transmission windows, especially around 500-540 nm where the green color filter becomes transparent. In RCWA (2D), the actual microlens profile focuses light, the Bayer color filter pattern introduces lateral index variation, and DTI boundaries confine light within the pixel. These 2D effects significantly increase the optical path length through silicon and reduce coherent back-reflection.

##### 2. Transmission

TMM predicts substantial transmission (5-91%) through the stack, with a dramatic peak near 520 nm (T=0.91). In contrast, both RCWA solvers show near-zero transmission (T < 0.015 for torcwa, T < 0.0004 for grcwa). The 3 um silicon layer at the Fourier order [3,3] resolution used in RCWA effectively absorbs all power that is not reflected. The S-matrix calculation through thick absorbing layers approaches machine precision limits, and lateral confinement by DTI prevents light from escaping sideways.

##### 3. Reflection

TMM reflectance exhibits strong thin-film oscillations (1-24%), characteristic of coherent interference in a planar multilayer. RCWA reflectance is low and spectrally smooth (0.1-2.6%). The 2D microlens profile acts as a graded-index anti-reflection structure: light incident on a curved surface couples more efficiently into the stack than light hitting a flat interface, reducing specular reflection significantly.

##### 4. torcwa vs grcwa Agreement

Both RCWA solvers agree well with each other, with reflectance differences within 0.5-1.5% absolute and absorption differences within 0.5-2%. This level of agreement between two independent RCWA implementations (PyTorch-based vs NumPy-based) provides strong validation that both solvers are computing the electromagnetic problem correctly. Small residual differences arise from numerical precision in eigendecomposition and S-matrix assembly.

### When to Use TMM vs RCWA

- **TMM**: Stack design, BARL thickness optimization, fast parameter sweeps (~3 ms per sweep). Best for thin-film interference analysis where lateral patterning is not the primary concern.
- **RCWA**: Full 2D diffraction effects, absolute QE prediction, cross-pixel crosstalk analysis, microlens design (~15 s per sweep). Required when lateral structure (microlens, Bayer pattern, DTI) significantly affects the result.

### Runtime Comparison

| Solver | Wavelengths | Runtime | Speedup |
|--------|:-----------:|--------:|--------:|
| TMM | 21 | 2.9 ms | 5400x |
| grcwa | 21 | 0.1 s | 157x |
| torcwa | 21 | 15.7 s | 1x |

TMM is approximately 5400x faster than torcwa, making it the preferred solver for iterative design loops. grcwa is 157x faster than torcwa for the same RCWA calculation, benefiting from NumPy's optimized linear algebra on CPU. torcwa's PyTorch backend is optimized for GPU acceleration, so the CPU-only comparison understates its performance on CUDA devices.

### Solver Compatibility Notes

::: warning meent Numerical Stability
meent 0.12.0 has a known numerical instability for multi-layer 2D structures: R+T > 1 occurs for stacks with 2 or more patterned layers. Single-layer simulations are correct. This is under investigation.
:::

::: warning FDTD Compatibility
The flaport fdtd 0.3.5 library has PML boundary API changes that require adapter updates. FDTD validation is planned for a future release.
:::

### Energy Conservation

Energy conservation (R + T + A = 1) is a fundamental physical constraint and serves as a self-consistency check for each solver.

| Solver | max \|1 - (R+T+A)\| | Notes |
|--------|:-------------------:|-------|
| TMM | 1.11 x 10^-16 | Machine precision (analytical transfer matrices) |
| torcwa | 0.0000 | S-matrix formulation inherently conserves energy |
| grcwa | 0.0000 | S-matrix formulation inherently conserves energy |

All three solvers satisfy energy conservation to machine precision, confirming that the numerical implementations are correct regardless of the physical differences in their modeling assumptions.

### Execution Environment

```
Platform    : macOS (Darwin 25.2.0, Apple Silicon)
Python      : 3.11
PyTorch     : 2.5.0 (torcwa backend)
NumPy       : (grcwa backend)
RCWA Order  : [3, 3] (49 harmonics)
Grid        : 64 x 64
Lens Slices : 5
```

## Part 3: RCWA vs FDTD

This section demonstrates how to validate RCWA (grcwa) results against FDTD (flaport) for a BSI CMOS pixel, and compare direct illumination with cone illumination.

### Interactive Chart

<RcwaFdtdValidation />

### Why Cross-Solver Validation?

RCWA and FDTD solve Maxwell's equations with fundamentally different approaches:

| | RCWA | FDTD |
|---|---|---|
| **Domain** | Frequency domain | Time domain |
| **Periodicity** | Inherently periodic | Requires PML boundaries |
| **Strengths** | Fast for thin-film stacks | Handles arbitrary geometry |
| **Convergence** | Fourier order | Grid spacing + runtime |

When both methods agree on absorption/reflection/transmission spectra, it provides strong confidence in the physical validity of the simulation.

### Running the Validation

```bash
PYTHONPATH=. python3.11 scripts/validate_rcwa_vs_fdtd.py
```

This script runs three experiments:

#### Experiment 1: Normal Incidence Sweep

Compares grcwa (fourier_order=[3,3]) vs fdtd_flaport (dx=0.02um, 300fs) across 400-700nm.

**Acceptance criterion:** max |A_grcwa - A_fdtd| < 10%

#### Experiment 2: Cone Illumination

Compares three illumination conditions using grcwa:
- **Direct**: Normal incidence (θ=0°)
- **Cone F/2.0 CRA=0°**: 19-point Fibonacci sampling, cosine weighting
- **Cone F/2.0 CRA=15°**: Same cone with 15° chief ray angle

#### Experiment 3: RCWA Cross-Check

Validates grcwa vs torcwa with identical cone illumination to ensure RCWA solver consistency.

**Acceptance criterion:** max |A_grcwa - A_torcwa| < 5%

### Using ConeIlluminationRunner

```python
from compass.runners.cone_runner import ConeIlluminationRunner

config = {
    "pixel": { ... },  # pixel stack config
    "solver": {"name": "grcwa", "type": "rcwa", "params": {"fourier_order": [5, 5]}},
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.02}},
        "angle": {"theta_deg": 0.0, "phi_deg": 0.0},
        "polarization": "unpolarized",
        "cone": {
            "cra_deg": 0.0,       # Chief ray angle
            "f_number": 2.0,       # Lens F-number
            "sampling": {"type": "fibonacci", "n_points": 19},
            "weighting": "cosine",
        },
    },
    "compute": {"backend": "cpu"},
}

result = ConeIlluminationRunner.run(config)
```

The runner:
1. Generates angular sampling points from `ConeIllumination`
2. For each angle: sets up source with (θ, φ) and runs the solver
3. Accumulates weighted R, T, A and per-pixel QE
4. Returns a single `SimulationResult` with the weighted average

### Convergence Tips

If RCWA and FDTD results diverge:

- **FDTD**: Increase `runtime` (300→500fs), decrease `grid_spacing` (0.02→0.01um), increase `pml_layers` (15→25)
- **RCWA**: Increase `fourier_order` ([3,3]→[5,5]→[9,9])

The two-pass reference normalization in FDTD is critical for accurate R/T extraction — it subtracts the incident field from the total field at the reflection detector.

## Part 4: Solver Selection Guide

Use the following decision tree to choose the right solver for your simulation:

### 1. Do you only have uniform (unpatterned) layers?

**Yes → Use TMM**
- ~1000x faster than RCWA, ~5000x faster than FDTD
- Ideal for BARL thickness optimization, stack design, fast parameter sweeps
- Runtime: ~3 ms for a full wavelength sweep (21 wavelengths)
- Limitation: Cannot model microlens focusing, Bayer pattern diffraction, or DTI confinement

### 2. Do you have periodic 2D patterned layers (microlens, color filter grid, DTI)?

**Yes → Use RCWA**

Choose the specific RCWA solver based on your needs:

| Solver | Backend | Key Strengths | Best For |
|--------|---------|---------------|----------|
| **torcwa** | PyTorch | GPU acceleration, S-matrix stability, TF32 disabled for precision | Production GPU runs, large Fourier orders |
| **grcwa** | NumPy | Autograd support, fast on CPU, critical for project | Inverse design, CPU-only environments, cross-validation |
| **meent** | PyTorch/JAX | Multi-backend flexibility | Experimental, single-layer structures only (see stability note) |
| **fmmax** | JAX | 4 FMM vector formulations | Research, advanced Fourier factorization studies |

### 3. Do you have non-periodic or complex 3D geometries?

**Yes → Use FDTD**

| Solver | Backend | Key Strengths | Best For |
|--------|---------|---------------|----------|
| **fdtd_flaport** | PyTorch | Simple API, differentiable | Quick FDTD prototyping |
| **fdtdz** | JAX | Efficient z-propagation | JAX-based workflows |
| **fdtdx** | JAX | Multi-GPU, differentiable | Large-scale inverse design |
| **meep** | C++/Python | Mature, feature-rich | Complex geometries, reference solutions |

### Quick Reference

| Scenario | Recommended Solver | Approximate Runtime |
|----------|-------------------|-------------------|
| BARL thickness sweep (1000 configs) | TMM | ~3 seconds total |
| Single wavelength QE (2D pixel) | grcwa or torcwa | 0.5-15 seconds |
| Full wavelength sweep (2D pixel) | grcwa (CPU) / torcwa (GPU) | 0.1-16 seconds |
| Inverse design optimization | grcwa (autograd) | Minutes per iteration |
| Cross-validation reference | Run TMM + 2 RCWA solvers | Compare results |
| Non-periodic structure | meep or fdtdx | Minutes to hours |
