---
title: Cross-Validation
description: Workflow for comparing two or more EM solvers on the same pixel structure using ComparisonRunner and SolverComparison in COMPASS.
---

# Cross-Validation

Cross-validation is the practice of running the same pixel structure through two or more independent EM solvers and comparing the results. It is one of the core motivations behind COMPASS -- establishing confidence that simulation predictions are correct by verifying agreement across different numerical methods.

## Why cross-validate

No single solver is guaranteed to be correct for all structures. Cross-validation helps you:

- Detect implementation bugs or configuration errors in any one solver
- Verify that RCWA Fourier order or FDTD grid spacing is converged
- Establish error bars on QE predictions
- Identify structures where different methods disagree (and investigate why)

Agreement within 1-2% absolute QE between well-converged solvers is expected for standard BSI pixel structures.

## Quick start with ComparisonRunner

The `ComparisonRunner` runs the same simulation config with multiple solvers and produces a comparison summary:

```python
from compass.runners.comparison_runner import ComparisonRunner

config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "bayer_map": [["R", "G"], ["G", "B"]],
        "layers": {
            "microlens": {"enabled": True, "height": 0.6,
                          "radius_x": 0.48, "radius_y": 0.48},
            "planarization": {"thickness": 0.3, "material": "sio2"},
            "color_filter": {
                "thickness": 0.6,
                "materials": {"R": "cf_red", "G": "cf_green", "B": "cf_blue"},
            },
            "barl": {"layers": [
                {"thickness": 0.010, "material": "sio2"},
                {"thickness": 0.025, "material": "hfo2"},
            ]},
            "silicon": {
                "thickness": 3.0, "material": "silicon",
                "dti": {"enabled": True, "width": 0.1},
            },
        },
    },
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01}},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}

solver_configs = [
    {
        "name": "torcwa", "type": "rcwa",
        "params": {"fourier_order": [9, 9]},
        "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
    },
    {
        "name": "grcwa", "type": "rcwa",
        "params": {"fourier_order": [9, 9], "dtype": "complex128"},
    },
    {
        "name": "meent", "type": "rcwa",
        "params": {"fourier_order": [9, 9], "backend": "numpy"},
    },
]

comparison = ComparisonRunner.run(config, solver_configs)
```

## Understanding the comparison output

`ComparisonRunner.run()` returns a dictionary with three keys:

```python
# Individual SimulationResult objects
results = comparison["results"]       # list of SimulationResult
labels = comparison["labels"]         # ["torcwa", "grcwa", "meent"]
summary = comparison["summary"]       # dict with comparison metrics

print("Max QE difference per pixel:")
for key, val in summary["max_qe_diff"].items():
    print(f"  {key}: {val:.4f}")

print("\nMean QE difference per pixel:")
for key, val in summary["mean_qe_diff"].items():
    print(f"  {key}: {val:.4f}")

print("\nMax relative error (%):")
for key, val in summary["max_qe_relative_error_pct"].items():
    print(f"  {key}: {val:.2f}%")

print("\nRuntime comparison:")
for key, val in summary["runtimes_seconds"].items():
    print(f"  {key}: {val:.2f} s")
```

## Using SolverComparison directly

For more detailed analysis, use the `SolverComparison` class directly:

```python
from compass.analysis.solver_comparison import SolverComparison

comp = SolverComparison(
    results=comparison["results"],
    labels=comparison["labels"],
    reference_idx=0,              # torcwa as reference
)

# Absolute QE difference vs reference for each pixel at each wavelength
qe_diff = comp.qe_difference()
for key, arr in qe_diff.items():
    print(f"{key}: max |dQE| = {arr.max():.5f}, mean = {arr.mean():.5f}")

# Relative error (%)
qe_rel = comp.qe_relative_error()
for key, arr in qe_rel.items():
    print(f"{key}: max relative error = {arr.max():.2f}%")

# Runtime comparison
runtimes = comp.runtime_comparison()
for solver, time_s in runtimes.items():
    print(f"{solver}: {time_s:.2f} s")
```

## Plotting the comparison

### QE spectrum overlay

```python
import numpy as np
import matplotlib.pyplot as plt
from compass.visualization.qe_plot import plot_qe_comparison

fig, (ax_main, ax_diff) = plt.subplots(
    2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
)

# Plot QE spectra on top panel
colors = {"torcwa": "tab:blue", "grcwa": "tab:orange", "meent": "tab:green"}
ref_result = comparison["results"][0]
wl_nm = ref_result.wavelengths * 1000

for result, label in zip(comparison["results"], comparison["labels"]):
    for pixel_name, qe in result.qe_per_pixel.items():
        if pixel_name.startswith("G"):
            ax_main.plot(wl_nm, qe, color=colors[label], label=f"{label} ({pixel_name})")
            break

ax_main.set_ylabel("Quantum Efficiency")
ax_main.set_title("Cross-Validation: Green QE Spectrum")
ax_main.legend()
ax_main.grid(True, alpha=0.3)

# Plot pairwise difference on bottom panel
for key, diff in comp.qe_difference().items():
    if "G_0_1" in key or "G_1_0" in key:
        ax_diff.plot(wl_nm, diff, label=key)

ax_diff.set_xlabel("Wavelength (nm)")
ax_diff.set_ylabel("|QE difference|")
ax_diff.legend()
ax_diff.grid(True, alpha=0.3)

plt.tight_layout()
```

### Agreement heatmap

```python
fig, ax = plt.subplots(figsize=(8, 6))

# Build matrix of max QE differences between all solver pairs
n = len(comparison["labels"])
diff_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        pair_comp = SolverComparison(
            results=[comparison["results"][i], comparison["results"][j]],
            labels=[comparison["labels"][i], comparison["labels"][j]],
        )
        all_diffs = pair_comp.qe_difference()
        max_diff = max(arr.max() for arr in all_diffs.values())
        diff_matrix[i, j] = max_diff
        diff_matrix[j, i] = max_diff

im = ax.imshow(diff_matrix, cmap="YlOrRd", vmin=0, vmax=0.05)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(comparison["labels"])
ax.set_yticklabels(comparison["labels"])
plt.colorbar(im, label="Max |QE diff|")
ax.set_title("Pairwise Max QE Difference")
```

<SolverComparisonChart />

## RCWA vs FDTD cross-validation

Comparing RCWA and FDTD provides the strongest validation because they use fundamentally different numerical methods:

```python
solver_configs = [
    {
        "name": "torcwa", "type": "rcwa",
        "params": {"fourier_order": [13, 13]},
        "stability": {"precision_strategy": "mixed"},
    },
    {
        "name": "fdtd_flaport", "type": "fdtd",
        "params": {"grid_spacing": 0.01, "runtime": 300, "pml_layers": 20},
    },
]

comparison = ComparisonRunner.run(config, solver_configs)
```

When comparing RCWA and FDTD, use high-accuracy settings for both (Fourier order >= 13 for RCWA, grid spacing <= 10 nm for FDTD) to ensure that any disagreement reflects a genuine difference in the methods rather than insufficient convergence.

## Interpreting discrepancies

| Max |dQE|   | Interpretation                                   |
|-------------|--------------------------------------------------|
| < 0.01      | Excellent agreement. Results are reliable.        |
| 0.01 - 0.03 | Good agreement. Normal for different solvers.     |
| 0.03 - 0.05 | Acceptable. Check convergence of both solvers.    |
| > 0.05      | Investigate. Possible convergence or config issue.|

Common causes of disagreement:

1. **Insufficient Fourier order** -- increase order and re-run
2. **Insufficient FDTD grid resolution** -- decrease grid spacing
3. **Different Fourier factorization rules** -- ensure both use Li's inverse rule
4. **Precision mismatch** -- one solver in float32, another in float64
5. **Boundary condition differences** -- periodic (RCWA) vs PML (FDTD)

## Command-line comparison

For batch processing, use the comparison script:

```bash
python scripts/compare_solvers.py experiment=solver_comparison

# Override solvers and pixel
python scripts/compare_solvers.py \
    experiment=solver_comparison \
    pixel=default_bsi_1um \
    solvers="[torcwa,grcwa]"
```

## Next steps

- [Running RCWA](./running-rcwa.md) -- RCWA solver details
- [Running FDTD](./running-fdtd.md) -- FDTD solver details
- [Solver Benchmark cookbook](../cookbook/solver-benchmark.md) -- full benchmark recipe
