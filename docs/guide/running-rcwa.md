---
title: Running RCWA Solvers
description: How to configure and run the torcwa, grcwa, and meent RCWA solvers in COMPASS, including solver-specific settings, stability controls, and convergence testing.
---

# Running RCWA Solvers

COMPASS provides three RCWA solver backends -- torcwa, grcwa, and meent -- all accessible through the same `SolverFactory` interface. This guide covers solver-specific configuration, stability settings, and convergence verification.

## Creating an RCWA solver

All RCWA solvers are created through `SolverFactory.create()`:

```python
from compass.solvers.base import SolverFactory

# torcwa (recommended default)
solver = SolverFactory.create("torcwa", solver_config, device="cuda")

# grcwa (NumPy/JAX backend, good for cross-validation)
solver = SolverFactory.create("grcwa", solver_config, device="cpu")

# meent (multi-backend: numpy, jax, torch)
solver = SolverFactory.create("meent", solver_config, device="cpu")
```

## Solver-specific configuration

### torcwa

torcwa is the primary RCWA solver. It uses PyTorch for GPU-accelerated matrix operations.

```yaml
solver:
  name: "torcwa"
  type: "rcwa"
  params:
    fourier_order: [9, 9]
    dtype: "complex64"
  stability:
    precision_strategy: "mixed"
    allow_tf32: false
    eigendecomp_device: "cpu"
    fourier_factorization: "li_inverse"
    eigenvalue_broadening: 1.0e-10
    energy_check:
      enabled: true
      tolerance: 0.02
      auto_retry_float64: true
```

Key considerations for torcwa:

- Always set `allow_tf32: false`. TF32 reduces mantissa precision from 23 to 10 bits on Ampere+ GPUs, which breaks S-matrix accuracy.
- The `"mixed"` precision strategy runs most computations in float32 but promotes eigendecomposition to float64 on CPU. This is the best speed/stability trade-off.
- Li's inverse rule (`fourier_factorization: "li_inverse"`) is critical for structures with high-contrast boundaries such as tungsten metal grids.

### grcwa

grcwa uses NumPy with optional JAX acceleration. It defaults to float64, making it more numerically stable but slower.

```yaml
solver:
  name: "grcwa"
  type: "rcwa"
  params:
    fourier_order: [9, 9]
    dtype: "complex128"
```

grcwa is best used for cross-validation against torcwa. Since it uses a different implementation of the same RCWA algorithm, agreement between the two confirms correctness.

### meent

meent supports three backends selectable at runtime:

```yaml
solver:
  name: "meent"
  type: "rcwa"
  params:
    fourier_order: [9, 9]
    dtype: "complex64"
    backend: "torch"   # "numpy" | "jax" | "torch"
```

The COMPASS adapter handles the unit conversion automatically -- meent uses nanometers internally while COMPASS uses micrometers.

## Stability settings

### PrecisionManager

Before running any RCWA solver, configure precision settings:

```python
from compass.solvers.rcwa.stability import PrecisionManager

PrecisionManager.configure(solver_config)
```

This disables TF32 and sets up the correct precision context. The `SingleRunner` calls this automatically, but direct solver usage should invoke it manually.

### Mixed precision eigendecomposition

The eigenvalue problem is the most numerically sensitive step in RCWA. Use mixed precision to keep speed while avoiding instability:

```python
from compass.solvers.rcwa.stability import PrecisionManager
import numpy as np

# NumPy path (grcwa, meent-numpy)
matrix = np.random.randn(722, 722) + 1j * np.random.randn(722, 722)
eigenvalues, eigenvectors = PrecisionManager.mixed_precision_eigen(matrix)

# PyTorch path (torcwa, meent-torch)
eigenvalues, eigenvectors = PrecisionManager.mixed_precision_eigen_torch(matrix_tensor)
```

Both functions promote the input to float64, perform eigendecomposition, then cast back to the original precision.

### Adaptive fallback

For production sweeps where some wavelengths may fail, use the `AdaptivePrecisionRunner`:

```python
from compass.solvers.rcwa.stability import AdaptivePrecisionRunner

runner = AdaptivePrecisionRunner(tolerance=0.02)
result = runner.run_with_fallback(solver, wavelength=0.45, config=solver_config)
```

The fallback chain is: GPU float32 -> GPU float64 -> CPU float64. If all three fail, the runner raises a `RuntimeError` suggesting you reduce the Fourier order.

## Convergence testing

Always verify that your results have converged before trusting them.

### Fourier order sweep

The Fourier order determines the number of harmonics. For order `[N, N]`, the matrix size is $(2N+1)^2$. Sweep `N` and check that QE stabilizes:

```python
import numpy as np
from compass.solvers.base import SolverFactory

orders = range(5, 22, 2)
green_qe_values = []

for N in orders:
    solver_config["params"]["fourier_order"] = [N, N]
    solver = SolverFactory.create("torcwa", solver_config)
    solver.setup_geometry(pixel_stack)
    solver.setup_source({"wavelength": 0.55, "theta": 0.0,
                         "phi": 0.0, "polarization": "unpolarized"})
    result = solver.run()

    green_qe = np.mean([
        qe for name, qe in result.qe_per_pixel.items()
        if name.startswith("G")
    ])
    green_qe_values.append(float(green_qe))
    print(f"Order {N:2d}: Green QE = {green_qe:.4f}")

# Check convergence: relative change between successive orders
for i in range(1, len(green_qe_values)):
    delta = abs(green_qe_values[i] - green_qe_values[i-1])
    converged = "CONVERGED" if delta < 0.005 else ""
    print(f"  Order {list(orders)[i]}: delta = {delta:.5f} {converged}")
```

Typical convergence: order [9, 9] is sufficient for most 1 um pitch pixels. Structures with fine metal grids or high-contrast layers may need [13, 13] or higher.

### Runtime vs accuracy trade-off

| Order   | Matrix size | Typical time (GPU) | Use case            |
|---------|-------------|---------------------|---------------------|
| [5, 5]  | 121         | 0.1 s               | Quick screening     |
| [9, 9]  | 361         | 0.3 s               | Standard production |
| [13,13] | 729         | 1.5 s               | High accuracy       |
| [17,17] | 1225        | 5.0 s               | Publication quality |

## Running a complete simulation

```python
from compass.runners.single_run import SingleRunner

config = {
    "pixel": { ... },  # pixel stack definition
    "solver": {
        "name": "torcwa",
        "type": "rcwa",
        "params": {"fourier_order": [9, 9]},
        "stability": {"precision_strategy": "mixed", "fourier_factorization": "li_inverse"},
    },
    "source": {
        "wavelength": {"mode": "sweep", "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01}},
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}

result = SingleRunner.run(config)
print(f"Completed: {len(result.wavelengths)} wavelengths")
```

## Next steps

- [Running FDTD](./running-fdtd.md) -- flaport FDTD solver configuration
- [Cross-validation](./cross-validation.md) -- compare RCWA solvers against each other
- [Troubleshooting](./troubleshooting.md) -- stability and convergence issues
