# COMPASS

[![CI](https://github.com/geniuskey/compass/actions/workflows/ci.yml/badge.svg)](https://github.com/geniuskey/compass/actions/workflows/ci.yml)
[![Docs](https://github.com/geniuskey/compass/actions/workflows/docs.yml/badge.svg)](https://geniuskey.github.io/compass/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Cross-solver Optical Modeling Platform for Advanced Sensor Simulation**

COMPASS is a Python framework for pixel-level quantum efficiency (QE) simulation
of CMOS image sensors. It provides a unified interface across multiple
electromagnetic solvers (RCWA and FDTD), enabling cross-validation, parametric
sweeps, and reproducible optical modeling of backside-illuminated (BSI) pixel
stacks.

## Features

- **Multi-solver support** -- nine solver backends (four RCWA, four FDTD, one TMM)
  behind one interface; swap solvers through a single configuration change
- **Parametric pixel modeling** -- define BSI pixel stacks (microlens, color filter,
  metal grid, photodiode) with Hydra-based YAML configs
- **Material database** -- built-in n/k data for Si, SiO2, SiN, W, Al, and common
  CFA resins with Cauchy/Sellmeier/tabulated interpolation
- **Cross-solver validation** -- run the same geometry on multiple solvers and
  compare QE, reflection, and field distributions
- **Visualization** -- publication-ready QE curves, field maps, and stack
  cross-section plots
- **Extensible architecture** -- add new solvers, materials, or analysis modules
  by implementing a base class

## Installation

### Quick install (core only)

```bash
pip install -e .
```

### Full install (all solvers + visualization)

```bash
pip install -e ".[all]"
```

### Solver group extras

```bash
pip install -e ".[rcwa]"   # torcwa, grcwa, meent, fmmax (+ jax)
pip install -e ".[fdtd]"   # flaport fdtd
pip install -e ".[viz]"    # pyvista, plotly
pip install -e ".[dev]"    # pytest, mypy, ruff
```

meep is conda-only (`conda install -c conda-forge pymeep`); fdtdz and fdtdx
follow their upstream install instructions.

## Quick Start

Run the default pixel stack with the lightweight TMM solver:

```bash
python scripts/run_simulation.py solver=tmm compute=cpu
```

Or run the same configuration from Python:

```python
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from compass.runners.single_run import SingleRunner

with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    cfg = compose(config_name="config", overrides=["solver=tmm", "compute=cpu"])

result = SingleRunner.run(OmegaConf.to_container(cfg, resolve=True))

print(result.wavelengths)     # wavelength grid in um
print(result.qe_per_pixel)    # dict of pixel QE arrays
print(result.reflection)      # spectral reflectance
```

## Solver Support

| Solver | Type | Backend | Notes |
|--------|------|---------|-------|
| [torcwa](https://github.com/kch3782/torcwa) | RCWA | PyTorch | Default backend; S-matrix, GPU |
| [grcwa](https://github.com/weiliangjinca/grcwa) | RCWA | autograd/NumPy | Cross-validation reference |
| [meent](https://github.com/kc-ml2/meent) | RCWA | NumPy/JAX/PyTorch | Multi-backend, analytic eigendecomp |
| [fmmax](https://github.com/facebookresearch/fmmax) | RCWA | JAX | 4 selectable vector formulations |
| [flaport/fdtd](https://github.com/flaport/fdtd) | FDTD | PyTorch | 2.5D FDTD, GPU + autograd |
| [fdtdz](https://github.com/spinsphotonics/fdtdz) | FDTD | JAX | 2D (z-invariant) cross-sections |
| [fdtdx](https://github.com/ymahlau/fdtdx) | FDTD | JAX | 3D, multi-GPU, differentiable |
| [meep](https://github.com/NanoComp/meep) | FDTD | C++/Python | Subpixel averaging, adjoint gradients |
| tmm | TMM | NumPy | 1D planar stacks, ~1000x faster than RCWA |

## Project Structure

```
compass/
  core/            # Types, config schema, units
  geometry/        # PixelStack, layer builders
  materials/       # Material database and interpolation
  solvers/         # RCWA / FDTD / TMM solver adapters
  sources/         # Planewave and cone illumination
  runners/         # Single-run, sweep, comparison, ROI orchestration
  analysis/        # QE extraction, crosstalk, energy balance
  optimization/    # Inverse design (scipy.optimize)
  diagnostics/     # RCWA stability diagnostics
  io/              # HDF5 / CSV / JSON export, ray files
  visualization/   # Plotting utilities
configs/           # Hydra YAML configurations
materials/         # n/k CSV data (metals, dielectrics, polymers, semiconductors)
scripts/           # CLI entry points, convergence studies, report generators
docs/              # VitePress documentation site
tests/             # Test suite
```

## Documentation

Full documentation is available at [geniuskey.github.io/compass](https://geniuskey.github.io/compass/)
or can be built locally:

```bash
cd docs && npm install && npm run dev
```

## Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes.
See the documentation site for development guidelines.

## License

MIT
