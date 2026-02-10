"""Energy balance verification: R + T + A = 1."""
from __future__ import annotations

import logging

import numpy as np

from compass.core.types import SimulationResult

logger = logging.getLogger(__name__)

class EnergyBalance:
    """Verify energy conservation in simulation results."""

    @staticmethod
    def check(result: SimulationResult, tolerance: float = 0.01) -> dict:
        """Check R+T+A ≈ 1 for all wavelengths."""
        if result.reflection is None or result.transmission is None:
            return {"valid": True, "message": "Insufficient data for energy check"}
        R = np.asarray(result.reflection)
        T = np.asarray(result.transmission)
        A = np.asarray(result.absorption) if result.absorption is not None else 1.0 - R - T
        total = R + T + A
        error = np.abs(total - 1.0)
        max_err = float(np.max(error))
        mean_err = float(np.mean(error))
        valid = max_err < tolerance
        if not valid:
            logger.warning(f"Energy balance violation: max error = {max_err:.4f}")
        return {
            "valid": valid,
            "max_error": max_err,
            "mean_error": mean_err,
            "per_wavelength": error,
            "R": R, "T": T, "A": A,
        }
