---
title: Changelog
description: Version history for the COMPASS simulation platform.
---

# Changelog

All notable changes to the COMPASS project are documented here. This project follows [Semantic Versioning](https://semver.org/).

## v0.1.0 -- Initial Release

**Release date**: 2025-02-10

The first public release of COMPASS, establishing the core simulation framework with multi-solver support, material database, and visualization tools.

### Core framework

- Pydantic-based configuration schema (`CompassConfig`) with full validation
- Hydra/OmegaConf integration for YAML config loading and CLI overrides
- Internal coordinate system: micrometers for all lengths, degrees for external angles, radians for internal angles
- Light propagation convention: -z direction (air at z_max, silicon at z_min)

### Geometry engine

- `PixelStack` and `GeometryBuilder` for solver-agnostic pixel structure definition
- Superellipse microlens profile with configurable height, radius, squareness, and CRA shift
- Multi-layer BARL (Bottom Anti-Reflective Layer) support
- Color filter with Bayer pattern (RGGB) and configurable metal grid
- Silicon layer with embedded photodiode and DTI (Deep Trench Isolation)
- Planarization (over-coat) layer
- Layer slice generation for RCWA (2D permittivity grids at each z-height)

### Solvers

- `SolverBase` abstract interface for all EM solvers
- `SolverFactory` with lazy import and plug-in registration
- **torcwa** RCWA solver adapter (PyTorch, CUDA GPU)
- **grcwa** RCWA solver adapter (NumPy/JAX)
- **meent** RCWA solver adapter (NumPy/JAX/PyTorch multi-backend)
- **fdtd_flaport** FDTD solver adapter (PyTorch, CUDA GPU)
- Unified `SimulationResult` output format across all solvers

### Numerical stability (5-layer defense)

- `PrecisionManager`: TF32 disable, mixed precision eigendecomposition
- `StableSMatrixAlgorithm`: Redheffer star product (no T-matrix divergence)
- `LiFactorization`: Li's inverse rule for high-contrast boundaries
- `EigenvalueStabilizer`: Degenerate eigenvalue handling, branch selection
- `AdaptivePrecisionRunner`: Automatic float32 -> float64 -> CPU fallback

### Sources

- `PlanewaveSource`: Single wavelength, sweep, and list modes
- TE, TM, and unpolarized (TE+TM average) polarization support
- `ConeIllumination`: Exit pupil illumination model with CRA, F-number, angular sampling (Fibonacci, grid), and weighting functions (uniform, cosine, cos4, Gaussian)

### Analysis

- `QECalculator`: Per-pixel quantum efficiency extraction
- `EnergyBalance`: R + T + A = 1 verification
- `SolverComparison`: Pairwise QE difference, relative error, runtime comparison

### Runners

- `SingleRunner`: Single simulation execution from config
- `SweepRunner`: Wavelength sweep orchestration
- `ComparisonRunner`: Multi-solver comparison workflow
- `ROISweepRunner`: Sensor-position sweep with CRA interpolation and microlens shift

### Visualization

- `plot_pixel_cross_section`: 2D cross-section views (XZ, YZ, XY planes)
- `plot_qe_spectrum`: QE vs wavelength per color channel
- `plot_qe_comparison`: Multi-result QE overlay with difference plot
- `plot_crosstalk_heatmap`: Cross-pixel energy distribution
- `field_plot_2d`: EM field distribution visualization
- 3D pixel stack viewer (PyVista/Plotly)

### I/O

- HDF5 result storage with embedded metadata and config
- CSV/JSON export for post-processing

### Materials database

- Built-in materials: air, silicon (Green 2008), SiO2, Si3N4, HfO2, TiO2, tungsten, polymer (n=1.56), color filter dyes (R/G/B)
- User-defined materials via CSV (wavelength, n, k)
- Analytical dispersion models: Sellmeier, Cauchy

### Infrastructure

- Python 3.10+ with type hints throughout
- pytest test suite with unit and integration tests
- ruff linting (E, F, W, I rules)
- mypy static type checking
- Google-style docstrings
