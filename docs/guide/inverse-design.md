# Inverse Design

COMPASS includes a built-in optimization module for inverse design of pixel structures. Starting from a baseline pixel configuration, the optimizer automatically searches for geometry parameters (microlens shape, BARL thicknesses, color filter thickness, etc.) that maximize quantum efficiency, minimize crosstalk, or achieve other user-defined targets.

## Overview

The optimization workflow has three parts:

1. **Parameter Space** -- Choose which physical dimensions to optimize and set their bounds.
2. **Objective Function** -- Define what "better" means: higher QE, lower crosstalk, or a weighted combination.
3. **Optimizer** -- Select an algorithm (Nelder-Mead, L-BFGS-B, differential evolution, etc.) that iteratively evaluates COMPASS simulations to find the optimum.

Under the hood, each optimizer iteration runs a full COMPASS simulation with updated parameters, evaluates the objective, and records the result in an optimization history.

## Quick Example

```python
import copy
from compass.optimization import (
    PixelOptimizer,
    ParameterSpace,
    MicrolensHeight,
    MicrolensSquareness,
    MaximizeQE,
    MinimizeCrosstalk,
    CompositeObjective,
)

# Start from your baseline config dict
base_config = {
    "pixel": {
        "pitch": 1.0,
        "unit_cell": [2, 2],
        "layers": {
            "microlens": {"height": 0.6, "profile": {"n": 2.5}},
            "color_filter": {"thickness": 0.6},
        },
    },
    "solver": {"name": "meent"},
    "source": {"wavelength": {"mode": "sweep", "sweep": {"start": 0.4, "stop": 0.7, "step": 0.02}}},
    "compute": {"backend": "cpu"},
}

# 1. Define parameter space
params = ParameterSpace([
    MicrolensHeight(base_config, min_val=0.2, max_val=1.2),
    MicrolensSquareness(base_config, min_val=1.5, max_val=5.0),
])

# 2. Define objective
objective = CompositeObjective([
    (1.0, MaximizeQE(wavelength_range=(0.4, 0.7))),
    (0.5, MinimizeCrosstalk()),
])

# 3. Run optimization
optimizer = PixelOptimizer(
    base_config=base_config,
    parameter_space=params,
    objective=objective,
    solver_name="meent",
    method="nelder-mead",
    max_iterations=50,
)
result = optimizer.optimize()

print(f"Best QE objective: {result.best_objective:.4f}")
print(f"Best parameters: {result.best_params}")
print(f"Evaluations: {result.n_evaluations}")
print(f"Converged: {result.converged}")

# Save optimization history
result.history.save("outputs/optimization_history.json")
```

## Using YAML Configuration

You can also drive optimization from a Hydra YAML config:

```yaml
# configs/experiment/optimize_microlens.yaml
optimization:
  solver: meent
  method: nelder-mead
  max_iterations: 50
  parameters:
    - type: microlens_height
      bounds: [0.2, 1.2]
    - type: microlens_squareness
      bounds: [1.5, 5.0]
  objective:
    type: composite
    components:
      - weight: 1.0
        type: maximize_qe
        wavelength_range: [0.4, 0.7]
      - weight: 0.5
        type: minimize_crosstalk
```

## Available Parameters

| Parameter | Class | Size | Description |
|-----------|-------|------|-------------|
| `microlens_height` | `MicrolensHeight` | 1 | Microlens sag height (um) |
| `microlens_squareness` | `MicrolensSquareness` | 1 | Superellipse exponent n |
| `microlens_radii` | `MicrolensRadii` | 2 | Lens semi-axes (radius_x, radius_y) |
| `barl_thicknesses` | `BARLThicknesses` | N | One value per BARL layer |
| `color_filter_thickness` | `ColorFilterThickness` | 1 | Color filter layer thickness (um) |

Each parameter class accepts `min_val` and `max_val` arguments for bound constraints.

## Objective Functions

All objectives return a scalar to be **minimized**. Maximization objectives (QE, peak QE) are internally negated.

| Objective | Description |
|-----------|-------------|
| `MaximizeQE` | Negative mean QE across pixels and wavelengths. Supports `target_pixels`, `wavelength_range`, and `weights`. |
| `MinimizeCrosstalk` | Mean off-diagonal QE fraction (crosstalk). Supports `target_wavelength_range`. |
| `MaximizePeakQE` | Negative peak QE for a specific color channel (`"R"`, `"G"`, or `"B"`). |
| `EnergyBalanceRegularizer` | Quadratic penalty when R+T+A deviates from 1 beyond a tolerance. |
| `CompositeObjective` | Weighted sum of multiple objectives. |

### CompositeObjective Example

```python
objective = CompositeObjective([
    (1.0, MaximizeQE()),
    (0.5, MinimizeCrosstalk()),
    (0.1, EnergyBalanceRegularizer(tolerance=0.02, penalty_weight=10.0)),
])
```

<EnergyBalanceDiagram />

## Optimization Methods

| Method | Type | Best For |
|--------|------|----------|
| `nelder-mead` | Gradient-free simplex | Robust default, 1--5 parameters |
| `powell` | Gradient-free directional | Smooth landscapes, bounded |
| `l-bfgs-b` | Quasi-Newton with bounds | Smooth, differentiable solvers |
| `differential-evolution` | Global stochastic | Multi-modal landscapes, avoids local minima |

## Tips

- **Start with Nelder-Mead** for up to ~5 parameters. It is robust and does not require gradients.
- **Use `differential-evolution`** when you suspect multiple local minima or have more than 5 parameters.
- **Add `EnergyBalanceRegularizer`** as a soft constraint to steer the optimizer away from unphysical configurations.
- **Save the history** after each run with `result.history.save(path)` so you can visualize convergence later.
- **Use a coarse wavelength sweep** during optimization (e.g. step=0.02 um) to speed up each evaluation, then verify with a fine sweep at the end.
- **Choose `meent`** as the solver for optimization since it is a fast, differentiable RCWA backend.
- Each evaluation runs a full COMPASS simulation, so wall time scales linearly with the number of evaluations.
