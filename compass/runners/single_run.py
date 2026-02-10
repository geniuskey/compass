"""Single simulation runner."""
from __future__ import annotations

import logging

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.materials.database import MaterialDB
from compass.solvers.base import SolverFactory

logger = logging.getLogger(__name__)

class SingleRunner:
    """Run a single simulation with given config."""

    @staticmethod
    def run(config: dict) -> SimulationResult:
        """Execute a single simulation from config dict."""
        _pixel_config = config.get("pixel", config)
        solver_config = config.get("solver", {})
        source_config = config.get("source", {})
        compute_config = config.get("compute", {})

        # Determine device
        device = compute_config.get("backend", "cpu")
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # Build pixel stack
        material_db = MaterialDB()
        pixel_stack = PixelStack(config, material_db)

        # Create solver
        solver_name = solver_config.get("name", "torcwa")
        solver = SolverFactory.create(solver_name, solver_config, device)

        # Setup and run
        solver.setup_geometry(pixel_stack)
        solver.setup_source(source_config)
        result = solver.run_timed()

        # Validate
        solver.validate_energy_balance(result)

        return result
