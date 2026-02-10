# COMPASS

**Cross-solver Optical Modeling Platform for Advanced Sensor Simulation**

COMPASS is a Python framework for pixel-level quantum efficiency (QE) simulation
of CMOS image sensors. It provides a unified interface across multiple
electromagnetic solvers (RCWA and FDTD), enabling cross-validation, parametric
sweeps, and reproducible optical modeling of backside-illuminated (BSI) pixel
stacks.

## Features

- **Multi-solver support** -- swap between torcwa, grcwa, meent, and flaport FDTD
  through a single configuration change
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

### Individual solver extras

```bash
pip install -e ".[torcwa]"
pip install -e ".[grcwa]"
pip install -e ".[meent]"
pip install -e ".[fdtd]"
```

## Quick Start

```python
import compass

# Load a default BSI 2x2 Bayer pixel config
sim = compass.load_config("configs/experiment/bsi_2x2_basic.yaml")

# Run with the torcwa RCWA solver
result = sim.run(solver="torcwa")

# Plot QE curves per color channel
result.plot_qe()

# Access raw data
print(result.qe_per_pixel)       # dict of pixel QE arrays
print(result.wavelengths)         # wavelength grid in um
print(result.reflection)          # spectral reflectance
```

## Solver Support

| Solver | Type | Backend | GPU | Status |
|--------|------|---------|-----|--------|
| [torcwa](https://github.com/kch3782/torcwa) | RCWA | PyTorch | Yes | Supported |
| [grcwa](https://github.com/weiliangjinca/grcwa) | RCWA | JAX/NumPy | Yes | Supported |
| [meent](https://github.com/kc-ml2/meent) | RCWA | PyTorch/JAX | Yes | Supported |
| [flaport/fdtd](https://github.com/flaport/fdtd) | FDTD | PyTorch | Yes | Supported |
| fdtdz | FDTD | JAX | Yes | Planned |
| meep | FDTD | C++/Python | No | Planned |

## Project Structure

```
compass/
  core/           # Types, config loading, constants
  geometry/        # PixelStack, layer builders
  materials/       # Material database and interpolation
  solvers/         # RCWA and FDTD solver adapters
  sources/         # Planewave and cone illumination
  analysis/        # QE extraction, field analysis
  visualization/   # Plotting utilities
configs/           # Hydra YAML configurations
docs/              # VitePress documentation site
tests/             # Test suite
```

## Documentation

Full documentation is available at [compass-sim.github.io/compass](https://compass-sim.github.io/compass)
or can be built locally:

```bash
cd docs && npm install && npm run dev
```

## Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes.
See the documentation site for development guidelines.

## License

MIT
