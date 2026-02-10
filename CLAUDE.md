# COMPASS - Cross-solver Optical Modeling Platform for Advanced Sensor Simulation

## Project Overview
- COMPASS simulates CMOS image sensor pixels using multiple EM solvers (RCWA: torcwa, grcwa, meent; FDTD: flaport)
- Single YAML config defines pixel structure → solver-agnostic PixelStack → multiple solver comparison
- Key outputs: QE per pixel per wavelength, crosstalk, field distributions

## Architecture
```
compass/
├── core/           # types.py (SimulationResult, LayerSlice, FieldData), config_schema.py, units.py
├── geometry/       # pixel_stack.py (PixelStack), builder.py (GeometryBuilder)
├── materials/      # database.py (MaterialDB) - built-in + CSV + Cauchy/Sellmeier
├── solvers/
│   ├── base.py     # SolverBase ABC, SolverFactory
│   ├── rcwa/       # torcwa_solver.py, grcwa_solver.py, meent_solver.py, stability.py
│   └── fdtd/       # flaport_solver.py
├── sources/        # planewave.py, cone_illumination.py, ray_file_reader.py
├── analysis/       # qe_calculator.py, energy_balance.py, solver_comparison.py
├── visualization/  # structure_plot_2d.py, field_plot_2d.py, qe_plot.py, viewer_3d.py
├── io/             # hdf5_handler.py, export.py
├── runners/        # single_run.py, sweep_runner.py, comparison_runner.py, roi_sweep_runner.py
└── diagnostics/    # stability_diagnostics.py
```

## Coordinate Conventions
- x, y: lateral (in-plane)
- z: stack direction (vertical)
- Light propagates in **-z** direction (air at z_max → silicon at z_min)
- Internal units: **micrometers (um)** for all lengths
- Angles: degrees (external), radians (internal)
- Permittivity: e = (n + ik)^2

## Code Conventions
- Python 3.10+, type hints required
- Pydantic for config validation
- numpy for arrays, torch for GPU tensors
- pytest for testing, ruff for linting
- Docstrings: Google style
- No TF32 in RCWA (torch.backends.cuda.matmul.allow_tf32 = False)

## Physical Constraints
- Energy conservation: R + T + A = 1 (tolerance < 1%)
- QE range: 0 <= QE <= 1
- Silicon n ~ 4.08, k ~ 0.028 at 550nm (Green 2008)
- Microlens: superellipse profile z(x,y) = h*(1-r^2)^(1/2a)
- RCWA stability: S-matrix only (no T-matrix), mixed precision eigendecomp

## Key Commands
```bash
# Run simulation
python scripts/run_simulation.py
python scripts/run_simulation.py solver=torcwa pixel=default_bsi_1um

# Run tests
PYTHONPATH=. python3.11 -m pytest tests/ -v

# Compare solvers
python scripts/compare_solvers.py experiment=solver_comparison
```

## File Ownership (Agent Teams)
- core-engine: compass/core/, compass/geometry/, compass/materials/, compass/config/
- solvers-physics: compass/solvers/, compass/sources/, compass/analysis/
- viz-io: compass/visualization/, compass/io/, docs/, notebooks/
- qa-benchmark: tests/, benchmarks/, compass/diagnostics/
