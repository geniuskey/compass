# Installation

## Requirements

- Python 3.10 or later
- A CUDA-capable GPU is recommended for RCWA solvers but not required

## Install from source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/compass-sim/compass.git
cd compass
pip install -e .
```

This installs the core package with base dependencies: numpy, torch, hydra-core, omegaconf, pydantic, h5py, matplotlib, scipy, pyyaml, and tqdm.

## Optional dependencies

COMPASS organizes solver and visualization backends as optional dependency groups. Install only what you need:

```bash
# RCWA solvers (torcwa, grcwa, meent)
pip install -e ".[rcwa]"

# FDTD solver (flaport)
pip install -e ".[fdtd]"

# 3D visualization (plotly, pyvista)
pip install -e ".[viz]"

# Everything
pip install -e ".[all]"

# Development tools (pytest, mypy, ruff)
pip install -e ".[dev]"
```

| Group   | Packages                | Use case                        |
|---------|-------------------------|---------------------------------|
| `rcwa`  | torcwa                  | RCWA solvers (torcwa, grcwa, meent) |
| `fdtd`  | fdtd                    | FDTD solver (flaport backend)   |
| `viz`   | pyvista, plotly          | Interactive 3D visualization    |
| `all`   | rcwa + fdtd + viz       | Full installation               |
| `dev`   | pytest, pytest-cov, mypy, ruff | Testing and linting        |

## CUDA setup

RCWA solvers benefit significantly from GPU acceleration via PyTorch CUDA. If you have an NVIDIA GPU:

1. Install a CUDA-compatible PyTorch build. Check [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command for your CUDA version:

```bash
# Example for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

2. Verify CUDA is available:

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))
```

3. COMPASS automatically detects the GPU when `compute.backend` is set to `"auto"` (the default). To force a specific device, set `compute.backend` in your config:

```yaml
compute:
  backend: "cuda"   # Force CUDA
  gpu_id: 0         # GPU index
```

### TF32 precision

COMPASS **disables TF32** for RCWA computations by default. TF32 reduces floating-point precision in matrix multiplications on Ampere+ GPUs, which can cause numerical instability in S-matrix calculations. This is controlled by:

```yaml
solver:
  stability:
    allow_tf32: false  # Default; do not change unless benchmarked
```

## Apple Silicon (MPS)

PyTorch supports Apple Metal Performance Shaders on M1/M2/M3 Macs. COMPASS can use MPS as a compute backend:

```yaml
compute:
  backend: "mps"
```

**Limitations:**
- Complex number support on MPS is incomplete in some PyTorch versions
- Eigendecomposition may fall back to CPU automatically
- Performance is generally slower than CUDA for RCWA workloads
- Test with `compute.backend: "cpu"` first if you encounter MPS-related errors

## CPU fallback

All solvers work on CPU without any GPU dependencies:

```yaml
compute:
  backend: "cpu"
  num_workers: 4
```

CPU mode is slower but fully functional. It is useful for debugging, small-pitch pixels, and CI/CD pipelines.

## Environment variables

| Variable              | Description                                  | Default |
|-----------------------|----------------------------------------------|---------|
| `COMPASS_MATERIALS`   | Path to custom materials directory           | `./materials/` |
| `COMPASS_OUTPUT_DIR`  | Default output directory for results         | `./outputs/` |
| `CUDA_VISIBLE_DEVICES`| Standard PyTorch GPU selection               | all GPUs |

## Verifying installation

Run the test suite to confirm everything is working:

```bash
# Run all tests
PYTHONPATH=. python3.11 -m pytest tests/ -v

# Run only unit tests (no GPU required)
PYTHONPATH=. python3.11 -m pytest tests/unit/ -v

# Skip slow benchmarks
PYTHONPATH=. python3.11 -m pytest tests/ -v -m "not slow"
```

Quick Python check:

```python
from compass.core.config_schema import CompassConfig
from compass.materials.database import MaterialDB
from compass.solvers.base import SolverFactory

# Config loads with defaults
config = CompassConfig()
print(f"Default solver: {config.solver.name}")

# Material database loads built-in materials
mat_db = MaterialDB()
print(f"Available materials: {mat_db.list_materials()}")

# Check which solvers are importable
print(f"Registered solvers: {SolverFactory.list_solvers()}")
```

If you see `ModuleNotFoundError` for a solver package, install the corresponding optional dependency group (see table above).
