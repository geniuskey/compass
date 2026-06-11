"""QE (Quantum Efficiency) calculation module."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class QECalculator:
    """Compute per-pixel QE from simulation results."""

    @staticmethod
    def from_absorption(
        absorption_per_pixel: dict[str, np.ndarray], incident_power: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Calculate QE from absorbed power in each photodiode.
        QE = P_absorbed_in_PD / P_incident
        """
        qe = {}
        for name, absorbed in absorption_per_pixel.items():
            qe[name] = np.clip(absorbed / np.maximum(incident_power, 1e-30), 0, 1)
        return qe

    @staticmethod
    def from_poynting_flux(
        flux_top: np.ndarray, flux_bottom: np.ndarray, incident_power: np.ndarray
    ) -> np.ndarray:
        """QE from Poynting vector flux difference at PD boundaries."""
        absorbed = flux_top - flux_bottom
        return np.asarray(np.clip(absorbed / np.maximum(incident_power, 1e-30), 0, 1))

    #: Nominal wavelength bands (um) assigned to each color channel for
    #: spectral crosstalk evaluation.
    CHANNEL_BANDS: dict[str, tuple[float, float]] = {
        "B": (0.40, 0.50),
        "G": (0.50, 0.60),
        "R": (0.60, 0.70),
    }

    @staticmethod
    def compute_crosstalk(
        qe_per_pixel: dict[str, np.ndarray],
        wavelengths: np.ndarray,
        bands: dict[str, tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Compute the band-integrated spectral crosstalk matrix.

        A single full-illumination simulation cannot tag photons by which
        pixel they were "intended" for, so crosstalk is defined spectrally:
        light inside color channel i's wavelength band is intended for pixels
        of that color.

            CT[i, j] = (signal collected by pixel j inside pixel i's band)
                       / (total signal collected inside pixel i's band)

        Rows sum to 1. Same-color entries represent correct collection;
        off-color entries are spectral crosstalk. Pixel color is taken from
        the pixel name prefix (e.g. "R_0_0" -> band of channel "R"); pixels
        with an unknown color prefix use the full simulated range.

        Args:
            qe_per_pixel: Mapping of pixel name to QE spectrum.
            wavelengths: Wavelength array (um) matching the QE spectra.
            bands: Optional override of the per-channel wavelength bands.

        Returns:
            Crosstalk matrix of shape (n_pixels, n_pixels).
        """
        pixels = sorted(qe_per_pixel.keys())
        n = len(pixels)
        wavelengths = np.asarray(wavelengths)
        bands = bands if bands is not None else QECalculator.CHANNEL_BANDS

        ct = np.zeros((n, n))
        for i, pi in enumerate(pixels):
            color = pi.split("_")[0]
            lo, hi = bands.get(color, (wavelengths.min(), wavelengths.max()))
            in_band = (wavelengths >= lo) & (wavelengths <= hi)
            if not np.any(in_band):
                in_band = np.ones_like(wavelengths, dtype=bool)
            band_signal = np.array([float(np.sum(qe_per_pixel[pj][in_band])) for pj in pixels])
            total = band_signal.sum()
            if total > 0:
                ct[i, :] = band_signal / total
            else:
                ct[i, i] = 1.0
        return ct

    @staticmethod
    def spectral_response(
        qe_per_pixel: dict[str, np.ndarray], wavelengths: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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
