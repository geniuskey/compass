# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COMPASS (Cross-solver Optical Modeling Platform for Advanced Sensor Simulation) simulates CMOS image sensor pixels using multiple electromagnetic solvers. A single YAML config defines a pixel structure → solver-agnostic PixelStack → runs on any registered solver → outputs QE per pixel per wavelength, crosstalk, and field distributions.

## Commands

```bash
# Run simulation (Hydra CLI)
python scripts/run_simulation.py
python scripts/run_simulation.py solver=torcwa pixel=default_bsi_1um source=wavelength_sweep

# Run all tests
PYTHONPATH=. python3.11 -m pytest tests/ -v

# Run single test file / single test
PYTHONPATH=. python3.11 -m pytest tests/unit/test_tmm.py -v
PYTHONPATH=. python3.11 -m pytest tests/unit/test_tmm.py::TestTMMSolverAdapter::test_full_run -v

# Lint and format
ruff check compass/ --fix
ruff format compass/

# Type checking
mypy compass/

# Docs (VitePress)
cd docs && npm install && npx vitepress dev    # dev server
cd docs && npx vitepress build                  # production build
```

## Architecture

### Core Pipeline

```
YAML Config → Pydantic validation → PixelStack (3D geometry)
                                         ↓
MaterialDB (n,k lookup) ──→ SolverBase.setup_geometry()
PlanewaveSource ──────────→ SolverBase.setup_source()
                                         ↓
                              SolverBase.run()
                                         ↓
                          SimulationResult (QE, R, T, A, fields)
                                         ↓
                          Analysis (QECalculator, EnergyBalance, SolverComparison)
```

### Solver System

**SolverBase** (`compass/solvers/base.py`): Abstract base class with 4 required methods:
- `setup_geometry(pixel_stack)` → configure solver mesh/layers
- `setup_source(source)` → set wavelength, angle, polarization
- `run()` → execute simulation, return `SimulationResult`
- `get_field_distribution()` → extract E/H field data

**SolverFactory**: Static registry with lazy imports. Solvers register at module load:
```python
SolverFactory.register("torcwa", TorcwaSolver)  # in torcwa_solver.py
solver = SolverFactory.create("torcwa", config, device)  # lazy import + instantiate
```

**Available solvers** (10 total):

| Name | Type | Module | Notes |
|------|------|--------|-------|
| torcwa | RCWA | `solvers/rcwa/torcwa_solver.py` | PyTorch, S-matrix, TF32 disabled |
| grcwa | RCWA | `solvers/rcwa/grcwa_solver.py` | JAX/NumPy, **critical — never remove** |
| meent | RCWA | `solvers/rcwa/meent_solver.py` | PyTorch/JAX |
| fmmax | RCWA | `solvers/rcwa/fmmax_solver.py` | JAX, 4 vector FMM formulations |
| fdtd_flaport | FDTD | `solvers/fdtd/flaport_solver.py` | PyTorch |
| fdtdz | FDTD | `solvers/fdtd/fdtdz_solver.py` | JAX |
| fdtdx | FDTD | `solvers/fdtd/fdtdx_solver.py` | JAX, multi-GPU, differentiable |
| meep | FDTD | `solvers/fdtd/meep_solver.py` | C++/Python |
| tmm | TMM | `solvers/tmm/tmm_solver.py` | Pure numpy, 1D planar only, ~1000x faster |

### Geometry System

**PixelStack** (`compass/geometry/pixel_stack.py`): Builds 3D pixel structure from config.
- Layer order bottom→top: silicon → BARL → color_filter → planarization → microlens → air
- z=0 at silicon bottom; light propagates in **-z** direction (air→silicon)
- `get_layer_slices(wavelength, nx, ny)` → list of `LayerSlice` for RCWA
- Caching: meshgrid, height map, DTI mask, metal grid mask

**GeometryBuilder** (`compass/geometry/builder.py`): Microlens superellipse profile, Bayer tiling, CRA shift.

### Config System (Hydra)

```
configs/
├── config.yaml              # defaults: pixel, solver, source, compute
├── pixel/                   # default_bsi_1um.yaml, default_bsi_0p8um.yaml
├── solver/                  # torcwa.yaml, grcwa.yaml, meent.yaml, fmmax.yaml, ...
├── source/                  # planewave.yaml, wavelength_sweep.yaml, cone_illumination.yaml
├── compute/                 # cuda.yaml, cpu.yaml, mps.yaml
└── experiment/              # solver_comparison.yaml, qe_benchmark.yaml, optimize_microlens.yaml
```

Override any parameter from CLI: `python scripts/run_simulation.py pixel.pitch=0.8 solver.params.fourier_order=[11,11]`

### Materials

**MaterialDB** (`compass/materials/database.py`): Supports constant, tabulated CSV (cubic spline interpolation), Cauchy, and Sellmeier models. 32+ built-in materials loaded from `materials/` CSV directory (metals, dielectrics, polymers, semiconductors). Epsilon cache keyed by `(name, rounded_wavelength)`.

### Optimization

`compass/optimization/`: Inverse design via scipy.optimize (Nelder-Mead, L-BFGS-B, Powell, DE). ObjectiveFunction ABC (MaximizeQE, MinimizeCrosstalk, etc.) + ParameterSpace (MicrolensHeight, BARLThicknesses, etc.).

### Docs (VitePress)

`docs/.vitepress/config.mts`: Navigation with `withMermaid()` wrapper. EN/KO i18n via locales. 35+ interactive Vue 3 components registered in `docs/.vitepress/theme/index.ts`. Components use `useLocale()` composable for i18n: `const { t } = useLocale(); t('English', '한국어')`.

## Conventions

### Physics
- Internal units: **micrometers (um)** for all lengths
- Angles: degrees (external API), radians (internal computation)
- Complex refractive index: n + ik; permittivity: ε = (n + ik)²
- Energy conservation: R + T + A = 1 (tolerance < 1%)
- QE range: 0 ≤ QE ≤ 1
- RCWA stability: S-matrix only (no T-matrix), mixed precision eigendecomp on CPU

### Code
- Python 3.10+, type hints required, Google-style docstrings
- Pydantic for config validation, numpy for arrays, torch for GPU tensors
- Ruff: line length 100, physics variable names allowed (R, T, A, E, etc. — see `ruff.toml` for ignore list)
- No TF32 in RCWA: `torch.backends.cuda.matmul.allow_tf32 = False`
- Solver errors: use `raise ValueError/RuntimeError`, not `assert`
- Optional solver imports: wrap with `contextlib.suppress(ImportError)` in `__init__.py`

### User Preferences
- Primary language: Korean
- Always push to remote when implementation is complete
- **grcwa is critical** — active project depends on it; never remove or deprecate
