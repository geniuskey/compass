"""Wavelength sweep runner."""
from __future__ import annotations
import logging
import numpy as np
from tqdm import tqdm
from compass.core.types import SimulationResult
from compass.runners.single_run import SingleRunner

logger = logging.getLogger(__name__)

class SweepRunner:
    """Run wavelength sweep simulations."""

    @staticmethod
    def run(config: dict) -> SimulationResult:
        """Execute wavelength sweep. Delegates to solver's internal sweep."""
        return SingleRunner.run(config)
