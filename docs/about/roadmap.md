---
title: Roadmap
description: Future development plans for the COMPASS simulation platform, including planned features, solver additions, and integration targets.
---

# Roadmap

This document outlines the planned development direction for COMPASS. Items are organized by priority and estimated timeline. Priorities may shift based on community feedback and research needs.

## Near-term (v0.2.0)

### Web UI

- Browser-based simulation setup and result visualization
- Real-time pixel stack editor with live cross-section preview
- Interactive QE spectrum and field distribution plots
- Job queue for long-running simulations
- Built on a Python backend (FastAPI) with a modern frontend framework

### Auto-optimization

- Gradient-free optimization of pixel parameters (microlens height, BARL thicknesses, DTI width)
- Objective functions: maximize broadband QE, minimize crosstalk, maximize color separation
- Bayesian optimization (Optuna) for efficient parameter search
- Constraint support: total stack height, fabrication design rules, material availability
- Multi-objective Pareto front for QE vs crosstalk trade-off

### Additional FDTD solvers

- **fdtdz**: JAX-based systolic GPU FDTD for extreme performance on TPU/GPU clusters
- **Meep**: MIT Electromagnetic Equation Propagation integration for the most mature open-source FDTD

### Enhanced cone illumination

- Ray file import from Zemax OpticStudio (ZRD format)
- Arbitrary pupil shapes (circular, rectangular, annular)
- Vignetting and pupil aberration modeling
- Per-field-point F-number variation

## Mid-term (v0.3.0)

### TCAD integration

- Interface with Sentaurus or similar TCAD tools for electrical simulation
- Export COMPASS optical generation profiles as input to drift-diffusion solvers
- Combined optical + electrical QE prediction
- Carrier diffusion model for approximate electrical crosstalk without full TCAD

### Advanced material models

- Drude-Lorentz dispersion for metals (broadband FDTD compatibility)
- Temperature-dependent optical constants
- Quantum dot and perovskite color filter materials
- Organic photodetector materials for flexible sensors
- Anisotropic materials (birefringent layers)

### Large unit cell support

- Efficient handling of 4x4, 6x6, 8x8, and 10x10 unit cells
- Memory-optimized RCWA for large matrices (Fourier order > 20)
- Domain decomposition for FDTD on multi-GPU systems
- Quad Bayer and nonacell (3x3) pattern support

### Batch processing and HPC

- Slurm/PBS job submission for cluster environments
- Distributed wavelength sweep across multiple nodes
- Checkpoint and restart for long sweeps
- Progress monitoring dashboard

## Long-term (v1.0.0)

### 3D CAD import

- GDSII layout import for real foundry pixel designs
- STL/OBJ mesh import for complex 3D structures
- Automatic voxelization for FDTD grid generation
- Layer-by-layer process flow definition

### ISP co-simulation

- Image Signal Processing pipeline integration
- Simulated raw Bayer image generation from COMPASS QE data
- Noise model (photon shot noise, read noise, dark current)
- Demosaic and color correction evaluation
- End-to-end image quality metrics (SNR, color accuracy, MTF)

### Machine learning surrogates

- Neural network surrogate models trained on COMPASS simulation data
- Real-time QE prediction for interactive design exploration
- Transfer learning across pixel geometries
- Physics-informed neural networks for extrapolation beyond training data

### Inverse design

- Topology optimization of pixel structures using adjoint methods
- Freeform microlens and metasurface design
- Fabrication-constrained optimization (minimum feature size, layer count limits)
- Multi-wavelength, multi-angle simultaneous optimization

### Additional solver backends

- Lumerical FDTD interface (commercial solver for validation)
- COMSOL Multiphysics FEM interface
- Custom FMM (Fourier Modal Method) implementation with analytic gradient support

## Community contributions

We welcome contributions in all areas. High-impact contribution opportunities:

- New solver adapters (implement `SolverBase` ABC)
- Material data (measured n,k spectra for sensor materials)
- Validation benchmarks (comparison against published data or commercial tools)
- Documentation and tutorials
- Performance optimization (GPU kernels, memory reduction)

See [Contributing](./contributing.md) for details on how to get started.
