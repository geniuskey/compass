---
title: Contributing
description: How to contribute to COMPASS, including development setup, adding solvers and materials, code style, and testing requirements.
---

# Contributing

Thank you for your interest in contributing to COMPASS. This guide covers how to set up a development environment, the contribution workflow, and guidelines for specific types of contributions.

## Development setup

### Prerequisites

- Python 3.10 or later
- Git
- (Optional) NVIDIA GPU with CUDA for GPU-accelerated solvers

### Clone and install

```bash
git clone https://github.com/compass-team/compass.git
cd compass

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all optional dependencies
pip install -e ".[all,dev]"
```

### Verify the installation

```bash
# Run the test suite
PYTHONPATH=. python -m pytest tests/ -v

# Run linting
ruff check compass/

# Run type checking
mypy compass/
```

## Code style

COMPASS follows these conventions:

- **Python 3.10+** with type hints on all public functions and methods
- **Pydantic** for configuration validation
- **Google-style docstrings** for all public classes and functions
- **ruff** for linting (rules: E, F, W, I)
- **Line length**: 100 characters maximum
- **Imports**: sorted by ruff (isort-compatible)

Example function with proper style:

```python
def compute_qe(
    result: SimulationResult,
    pixel_stack: PixelStack,
    wavelength: float,
) -> dict[str, float]:
    """Compute quantum efficiency per pixel from simulation result.

    Extracts the absorbed power in each photodiode region and normalizes
    by the incident power to obtain QE.

    Args:
        result: Completed simulation result with field data.
        pixel_stack: Pixel geometry for photodiode region identification.
        wavelength: Wavelength in micrometers.

    Returns:
        Dictionary mapping pixel names (e.g., "R_0_0") to QE values.

    Raises:
        ValueError: If result does not contain absorption data.
    """
    ...
```

## Adding a new solver

COMPASS uses a plug-in architecture for solvers. To add a new solver:

### 1. Create the solver module

Create a new file under the appropriate directory:

- RCWA solvers: `compass/solvers/rcwa/your_solver.py`
- FDTD solvers: `compass/solvers/fdtd/your_solver.py`

### 2. Implement the SolverBase interface

```python
from compass.solvers.base import SolverBase, SolverFactory
from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
import numpy as np


class YourSolver(SolverBase):
    """Your solver adapter for COMPASS."""

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        """Convert PixelStack to your solver's geometry format."""
        self._pixel_stack = pixel_stack
        # Convert layers, materials, and geometry to your solver's format
        ...

    def setup_source(self, source_config: dict) -> None:
        """Configure the excitation source."""
        self._source_config = source_config
        # Set wavelength, angle, polarization
        ...

    def run(self) -> SimulationResult:
        """Execute the simulation."""
        # Run your solver
        # Extract R, T, A, QE per pixel, field data
        # Return standardized SimulationResult
        ...

    def get_field_distribution(
        self, component: str, plane: str, position: float
    ) -> np.ndarray:
        """Extract a 2D field slice."""
        ...


# Register with the factory
SolverFactory.register("your_solver", YourSolver)
```

### 3. Add to the import map

In `compass/solvers/base.py`, add your solver to the `_try_import` map:

```python
import_map = {
    ...
    "your_solver": "compass.solvers.rcwa.your_solver",  # or fdtd
}
```

### 4. Add tests

Create `tests/test_your_solver.py` with at minimum:

- Solver creation test
- Geometry setup test
- Single-wavelength run test
- Energy balance validation test

## Adding new materials

### Built-in material (CSV)

Add a CSV file under `materials/` with columns: `wavelength_um`, `n`, `k`:

```csv
wavelength_um,n,k
0.380,1.520,0.000
0.400,1.518,0.000
0.420,1.516,0.000
...
```

Then register it in the material database configuration.

### Analytical dispersion model

For materials defined by Sellmeier or Cauchy equations, add the coefficients to the material database YAML:

```yaml
your_material:
  model: "sellmeier"
  coefficients:
    B: [1.0396, 0.2318, 1.0105]
    C: [0.00600, 0.02002, 103.56]
```

## Testing

### Running tests

```bash
# All tests
PYTHONPATH=. python -m pytest tests/ -v

# Specific test file
PYTHONPATH=. python -m pytest tests/test_geometry.py -v

# Skip slow tests (FDTD, large sweeps)
PYTHONPATH=. python -m pytest tests/ -v -m "not slow"

# With coverage
PYTHONPATH=. python -m pytest tests/ --cov=compass --cov-report=html
```

### Writing tests

- Place tests in `tests/` with the naming convention `test_<module>.py`
- Use pytest fixtures for shared setup (configs, material database, pixel stacks)
- Mark slow tests with `@pytest.mark.slow`
- Test physical constraints: QE in [0, 1], energy balance R + T + A = 1

Example test:

```python
import numpy as np
import pytest
from compass.solvers.base import SolverFactory


def test_solver_energy_balance(pixel_stack, solver_config):
    """Verify that R + T + A = 1 within tolerance."""
    solver = SolverFactory.create("torcwa", solver_config)
    solver.setup_geometry(pixel_stack)
    solver.setup_source({"wavelength": 0.55, "theta": 0.0,
                         "phi": 0.0, "polarization": "unpolarized"})
    result = solver.run()

    total = result.reflection + result.transmission + result.absorption
    assert np.allclose(total, 1.0, atol=0.02), (
        f"Energy not conserved: R+T+A = {total}"
    )
```

## Submitting changes

1. Fork the repository and create a feature branch from `main`
2. Make your changes following the code style guidelines above
3. Add or update tests to cover your changes
4. Run the full test suite and linter to verify nothing is broken
5. Write a clear commit message describing the change
6. Open a pull request with a description of what the change does and why

### Commit message format

Use a concise imperative-style subject line:

```
Add meep FDTD solver adapter

Implement SolverBase interface for MIT Meep, supporting:
- 3D simulation with periodic/PML boundaries
- Broadband source with DFT monitors
- Field extraction on arbitrary planes
```

## Getting help

- Open an issue for bug reports or feature requests
- Use discussions for questions about usage or design decisions
- Tag issues with appropriate labels: `bug`, `enhancement`, `solver`, `material`, `docs`
