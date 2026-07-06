"""Single simulation runner."""
from __future__ import annotations

import copy
import logging

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.materials.database import MaterialDB
from compass.solvers.base import SolverFactory

logger = logging.getLogger(__name__)

# dtype promotion used when energy conservation fails and
# stability.energy_check.auto_retry_float64 is enabled.
_RETRY_DTYPES = {"complex64": "complex128", "float32": "float64"}

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

        # Energy conservation check (solver.stability.energy_check)
        energy_cfg = solver_config.get("stability", {}).get("energy_check", {})
        if not energy_cfg.get("enabled", True):
            return result
        tolerance = energy_cfg.get("tolerance", 0.01)
        if solver.validate_energy_balance(result, tolerance=tolerance):
            return result

        # Optional retry at higher precision when the check fails
        dtype = solver_config.get("params", {}).get("dtype")
        retry_dtype = _RETRY_DTYPES.get(dtype)
        if not energy_cfg.get("auto_retry_float64", False) or retry_dtype is None:
            return result

        logger.warning(
            f"{solver_name}: energy conservation violated with dtype={dtype}; "
            f"retrying with dtype={retry_dtype}"
        )
        retry_config = copy.deepcopy(solver_config)
        retry_config["params"]["dtype"] = retry_dtype
        solver = SolverFactory.create(solver_name, retry_config, device)
        solver.setup_geometry(pixel_stack)
        solver.setup_source(source_config)
        result = solver.run_timed()
        result.metadata["energy_retry_dtype"] = retry_dtype
        solver.validate_energy_balance(result, tolerance=tolerance)

        return result
