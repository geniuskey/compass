# Solver Benchmark

This recipe compares RCWA solvers (torcwa, grcwa, meent) on the same pixel structure, measuring accuracy and performance.

## Goal

Run the same BSI pixel simulation with all available RCWA solvers and one FDTD solver, then compare QE results and execution times.

## Setup

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

## Run all RCWA solvers

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

## Compare QE spectra

```python
if len(results) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_qe_comparison(results, labels, ax=ax)
    ax.set_title("RCWA Solver Comparison")
    plt.tight_layout()
    plt.savefig("solver_comparison_qe.png", dpi=150)
    plt.show()
```

## Quantitative comparison

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

## Fourier order convergence comparison

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

## Runtime scaling

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

## Expected results

- **QE agreement**: All RCWA solvers should agree within 0.5-1% QE at the same Fourier order.
- **Convergence**: All solvers converge to the same QE as order increases. Convergence rate may differ slightly.
- **Runtime**: torcwa is typically fastest on GPU. grcwa and meent are competitive.
- **FDTD vs RCWA**: Should agree within 1-2% QE. FDTD is significantly slower for single-wavelength runs.
