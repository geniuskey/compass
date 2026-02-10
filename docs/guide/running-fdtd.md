---
title: Running FDTD Solvers
description: How to configure and run the flaport FDTD solver in COMPASS, including grid spacing, runtime, PML settings, and broadband versus single-wavelength operation.
---

# Running FDTD Solvers

COMPASS supports FDTD (Finite-Difference Time-Domain) simulation through the flaport fdtd backend. FDTD is complementary to RCWA -- it is broadband in a single run, handles arbitrary geometries, and provides full time-domain field evolution, but is slower and more memory-intensive.

## When to use FDTD

Use FDTD when you need:

- Broadband spectral response from a single simulation run
- Non-periodic or finite-sized structures
- Near-field distribution detail inside the pixel stack
- Cross-validation against RCWA results
- Time-domain analysis of field propagation

For standard periodic Bayer QE sweeps, RCWA (torcwa) is faster. See [Choosing a Solver](./choosing-solver.md) for a detailed comparison.

## Creating the FDTD solver

```python
from compass.solvers.base import SolverFactory

solver_config = {
    "name": "fdtd_flaport",
    "type": "fdtd",
    "params": {
        "grid_spacing": 0.02,   # 20 nm grid cells
        "runtime": 200,          # 200 femtoseconds
        "pml_layers": 15,        # 15-cell PML absorber
        "dtype": "float64",
    },
}

solver = SolverFactory.create("fdtd_flaport", solver_config, device="cuda")
```

## Grid spacing

The grid spacing (`grid_spacing` in micrometers) is the most important FDTD parameter. It must resolve both the smallest geometric feature and the shortest wavelength inside the highest-index material.

The rule of thumb for visible-range silicon simulations:

$$\Delta x \leq \frac{\lambda_{\min}}{n_{\max} \times \text{PPW}}$$

where PPW (points per wavelength) should be at least 10. For silicon ($n \approx 4.0$) at 400 nm:

$$\Delta x \leq \frac{0.400}{4.0 \times 10} = 0.010 \text{ um (10 nm)}$$

In practice, 20 nm is adequate for most visible-range simulations and provides a good trade-off between accuracy and resource usage.

### Grid spacing recommendations

| Grid spacing | PPW at 400nm/Si | Memory (2x2, 1um) | Runtime | Accuracy  |
|-------------|-----------------|---------------------|---------|-----------|
| 40 nm       | 2.5             | ~250 MB             | ~15 s   | Low       |
| 20 nm       | 5.0             | ~2 GB               | ~45 s   | Adequate  |
| 10 nm       | 10.0            | ~8 GB               | ~180 s  | High      |
| 5 nm        | 20.0            | ~64 GB              | ~900 s  | Very high |

Memory scales as $O(1/\Delta x^3)$ in 3D, so halving the grid spacing increases memory by 8x.

## Runtime configuration

The `runtime` parameter (in femtoseconds) controls how long the FDTD simulation runs. The simulation must run long enough for:

1. The source pulse to propagate through the entire structure
2. All internal reflections to decay sufficiently
3. The fields at the monitors to reach steady state

```yaml
solver:
  name: "fdtd_flaport"
  type: "fdtd"
  params:
    grid_spacing: 0.02
    runtime: 200          # femtoseconds
    pml_layers: 15
    dtype: "float64"
```

A runtime of 200 fs is sufficient for most BSI pixel structures with 3 um silicon. For thicker silicon (>4 um) or structures with high-Q resonances, increase to 300-500 fs.

You can verify convergence by comparing results at different runtimes:

```python
import copy
from compass.runners.single_run import SingleRunner

runtimes = [100, 150, 200, 300]

for rt in runtimes:
    cfg = copy.deepcopy(config)
    cfg["solver"]["params"]["runtime"] = rt
    result = SingleRunner.run(cfg)
    avg_qe = sum(qe.mean() for qe in result.qe_per_pixel.values()) / len(result.qe_per_pixel)
    print(f"Runtime {rt} fs: avg QE = {avg_qe:.4f}")
```

## PML configuration

PML (Perfectly Matched Layer) absorbs outgoing radiation at the simulation boundaries. The `pml_layers` parameter sets the thickness of the PML region in grid cells.

```python
solver_config = {
    "name": "fdtd_flaport",
    "type": "fdtd",
    "params": {
        "grid_spacing": 0.02,
        "runtime": 200,
        "pml_layers": 15,    # 15 cells = 0.3 um at 20 nm grid
        "dtype": "float64",
    },
}
```

Guidelines:

- **15 cells** is the default and works well for most simulations
- Increase to **20-25 cells** if you observe spurious reflections from the boundaries
- For periodic structures, PML is applied only in the z-direction (top/bottom); lateral boundaries use periodic boundary conditions
- PML adds to the simulation domain size, so thicker PML increases memory usage

## Broadband vs single-wavelength

### Broadband operation

FDTD naturally produces broadband results from a single simulation because the source is a time-domain pulse. After the simulation, a Fourier transform of the recorded fields yields the spectral response at all frequencies simultaneously.

```python
config = {
    "pixel": { ... },
    "solver": {
        "name": "fdtd_flaport",
        "type": "fdtd",
        "params": {"grid_spacing": 0.02, "runtime": 200, "pml_layers": 15},
    },
    "source": {
        "wavelength": {
            "mode": "sweep",
            "sweep": {"start": 0.40, "stop": 0.70, "step": 0.01},
        },
        "polarization": "unpolarized",
    },
    "compute": {"backend": "auto"},
}

result = SingleRunner.run(config)
# All 31 wavelength points come from a single FDTD run
print(f"Wavelengths: {len(result.wavelengths)} points")
```

This is the key advantage of FDTD for dense spectral sampling. A 100-point wavelength sweep takes roughly the same time as a 10-point sweep, unlike RCWA where each wavelength is a separate computation.

### Single-wavelength operation

For a single wavelength, FDTD uses a CW (continuous wave) source. This is slower than RCWA for a single point but provides the full time-domain field evolution:

```python
config["source"] = {
    "wavelength": {"mode": "single", "value": 0.55},
    "polarization": "unpolarized",
}

result = SingleRunner.run(config)
```

## Grid spacing convergence test

Always verify convergence by running at multiple grid spacings:

```python
import copy
import numpy as np
from compass.runners.single_run import SingleRunner

spacings = [0.04, 0.02, 0.01]
qe_results = []

for dx in spacings:
    cfg = copy.deepcopy(config)
    cfg["solver"]["params"]["grid_spacing"] = dx
    result = SingleRunner.run(cfg)
    green_qe = np.mean([
        qe for name, qe in result.qe_per_pixel.items()
        if name.startswith("G")
    ])
    qe_results.append(green_qe)
    print(f"Grid {dx*1000:.0f} nm: Green QE = {green_qe:.4f}")

# Check convergence
for i in range(1, len(qe_results)):
    delta = abs(qe_results[i] - qe_results[i-1])
    print(f"  {spacings[i]*1000:.0f} nm vs {spacings[i-1]*1000:.0f} nm: "
          f"delta = {delta:.5f}")
```

## GPU and memory considerations

The flaport FDTD solver uses PyTorch tensors and supports CUDA GPU acceleration. Key points:

- **GPU memory**: A 2x2 unit cell at 20 nm grid spacing requires approximately 2 GB of VRAM. At 10 nm, this increases to about 8 GB.
- **CPU fallback**: If GPU memory is insufficient, set `backend: "cpu"`. CPU execution is 5-10x slower but has no memory ceiling beyond system RAM.
- **dtype**: Use `float64` for accuracy. `float32` saves memory but may introduce artifacts in the Fourier transform of long time traces.

```yaml
compute:
  backend: "cuda"    # or "cpu" for memory-constrained systems
  gpu_id: 0
```

## Energy balance validation

After every FDTD run, check energy conservation:

```python
from compass.analysis.energy_balance import EnergyBalance

check = EnergyBalance.check(result, tolerance=0.02)
print(f"Valid: {check['valid']}, max error: {check['max_error']:.4f}")
```

FDTD energy balance errors are typically larger than RCWA (1-3% vs <1%) due to the spatial discretization. If errors exceed 3%, reduce the grid spacing.

## Next steps

- [Running RCWA](./running-rcwa.md) -- RCWA solver configuration
- [Cross-validation](./cross-validation.md) -- compare FDTD results against RCWA
- [Choosing a Solver](./choosing-solver.md) -- when to use FDTD vs RCWA
