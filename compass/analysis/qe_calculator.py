"""QE (Quantum Efficiency) calculation module."""
from __future__ import annotations
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from compass.core.types import SimulationResult

logger = logging.getLogger(__name__)

class QECalculator:
    """Compute per-pixel QE from simulation results."""

    @staticmethod
    def from_absorption(absorption_per_pixel: Dict[str, np.ndarray], incident_power: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate QE from absorbed power in each photodiode.
        QE = P_absorbed_in_PD / P_incident
        """
        qe = {}
        for name, absorbed in absorption_per_pixel.items():
            qe[name] = np.clip(absorbed / np.maximum(incident_power, 1e-30), 0, 1)
        return qe

    @staticmethod
    def from_poynting_flux(flux_top: np.ndarray, flux_bottom: np.ndarray, incident_power: np.ndarray) -> np.ndarray:
        """QE from Poynting vector flux difference at PD boundaries."""
        absorbed = flux_top - flux_bottom
        return np.clip(absorbed / np.maximum(incident_power, 1e-30), 0, 1)

    @staticmethod
    def compute_crosstalk(qe_per_pixel: Dict[str, np.ndarray], bayer_map: list) -> np.ndarray:
        """Compute crosstalk matrix.
        Crosstalk(i,j) = fraction of light intended for pixel i that ends up in pixel j.
        """
        pixels = sorted(qe_per_pixel.keys())
        n = len(pixels)
        n_wl = len(next(iter(qe_per_pixel.values())))
        ct = np.zeros((n, n, n_wl))
        for i, pi in enumerate(pixels):
            total = sum(qe_per_pixel[pj] for pj in pixels)
            for j, pj in enumerate(pixels):
                ct[i, j, :] = qe_per_pixel[pj] / np.maximum(total, 1e-30)
        return ct

    @staticmethod
    def spectral_response(qe_per_pixel: Dict[str, np.ndarray], wavelengths: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Group QE by color channel (average over same-color pixels)."""
        color_qe = {}
        color_count = {}
        for name, qe in qe_per_pixel.items():
            color = name.split("_")[0]
            if color not in color_qe:
                color_qe[color] = np.zeros_like(qe)
                color_count[color] = 0
            color_qe[color] += qe
            color_count[color] += 1
        return {c: (wavelengths, color_qe[c] / color_count[c]) for c in sorted(color_qe)}
