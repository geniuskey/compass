# RCWA vs FDTD Cross-Solver Validation

This recipe demonstrates how to validate RCWA (grcwa) results against FDTD (flaport) for a BSI CMOS pixel, and compare direct illumination with cone illumination.

## Interactive Chart

<RcwaFdtdValidation />

## Why Cross-Solver Validation?

RCWA and FDTD solve Maxwell's equations with fundamentally different approaches:

| | RCWA | FDTD |
|---|---|---|
| **Domain** | Frequency domain | Time domain |
| **Periodicity** | Inherently periodic | Requires PML boundaries |
| **Strengths** | Fast for thin-film stacks | Handles arbitrary geometry |
| **Convergence** | Fourier order | Grid spacing + runtime |

When both methods agree on absorption/reflection/transmission spectra, it provides strong confidence in the physical validity of the simulation.

## Running the Validation

```bash
PYTHONPATH=. python3.11 scripts/validate_rcwa_vs_fdtd.py
```

This script runs three experiments:

### Experiment 1: Normal Incidence Sweep

Compares grcwa (fourier_order=[3,3]) vs fdtd_flaport (dx=0.02um, 300fs) across 400-700nm.

**Acceptance criterion:** max |A_grcwa - A_fdtd| < 10%

### Experiment 2: Cone Illumination

Compares three illumination conditions using grcwa:
- **Direct**: Normal incidence (θ=0°)
- **Cone F/2.0 CRA=0°**: 19-point Fibonacci sampling, cosine weighting
- **Cone F/2.0 CRA=15°**: Same cone with 15° chief ray angle

### Experiment 3: RCWA Cross-Check

Validates grcwa vs torcwa with identical cone illumination to ensure RCWA solver consistency.

**Acceptance criterion:** max |A_grcwa - A_torcwa| < 5%

## Using ConeIlluminationRunner

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

## Convergence Tips

If RCWA and FDTD results diverge:

- **FDTD**: Increase `runtime` (300→500fs), decrease `grid_spacing` (0.02→0.01um), increase `pml_layers` (15→25)
- **RCWA**: Increase `fourier_order` ([3,3]→[5,5]→[9,9])

The two-pass reference normalization in FDTD is critical for accurate R/T extraction — it subtracts the incident field from the total field at the reflection detector.
