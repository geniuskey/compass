"""Dedicated crosstalk analysis module for COMPASS."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from compass.core.types import SimulationResult

logger = logging.getLogger(__name__)


class CrosstalkAnalyzer:
    """Analyse pixel-to-pixel crosstalk from QE simulation data.

    Crosstalk quantifies the fraction of light intended for one pixel that
    is instead collected by a neighbouring pixel.  This class provides
    methods to build and interrogate the full crosstalk matrix and to
    extract summary statistics suitable for automated reporting.
    """

    @staticmethod
    def compute_matrix(
        qe_per_pixel: Dict[str, np.ndarray],
        bayer_map: list,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Compute the 3D crosstalk matrix.

        Crosstalk(i, j, k) is the fraction of light intended for pixel *i*
        that ends up in pixel *j* at wavelength index *k*.

        Args:
            qe_per_pixel: Mapping of pixel name to QE spectrum array.
            bayer_map: Bayer pattern specification (used for pixel ordering).
            wavelengths: Wavelength array in um.

        Returns:
            3D numpy array of shape (n_pixels, n_pixels, n_wavelengths).
        """
        pixels = sorted(qe_per_pixel.keys())
        n_pixels = len(pixels)
        n_wl = len(wavelengths)
        ct = np.zeros((n_pixels, n_pixels, n_wl))

        for i, pi in enumerate(pixels):
            # Total signal collected across all pixels when pixel i is
            # the intended target.
            total = np.zeros(n_wl)
            for pj in pixels:
                total += qe_per_pixel[pj]
            safe_total = np.maximum(total, 1e-30)
            for j, pj in enumerate(pixels):
                ct[i, j, :] = qe_per_pixel[pj] / safe_total

        return ct

    @staticmethod
    def spectral_crosstalk(
        ct_matrix: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Per-wavelength average crosstalk.

        For each wavelength, the average off-diagonal element of the
        crosstalk matrix is returned.

        Args:
            ct_matrix: 3D crosstalk matrix (n_pixels x n_pixels x n_wl).
            wavelengths: Wavelength array in um (length n_wl).

        Returns:
            1D array of length n_wl with average off-diagonal crosstalk
            at each wavelength.
        """
        n_pixels = ct_matrix.shape[0]
        n_wl = ct_matrix.shape[2]
        avg_ct = np.zeros(n_wl)

        if n_pixels < 2:
            return avg_ct

        for k in range(n_wl):
            mat = ct_matrix[:, :, k]
            off_diag_sum = np.sum(mat) - np.trace(mat)
            n_off_diag = n_pixels * n_pixels - n_pixels
            avg_ct[k] = off_diag_sum / max(n_off_diag, 1)

        return avg_ct

    @staticmethod
    def peak_crosstalk(
        ct_matrix: np.ndarray,
        wavelengths: np.ndarray,
    ) -> Tuple[float, float]:
        """Find peak crosstalk value and corresponding wavelength.

        The peak is defined as the maximum average off-diagonal crosstalk
        across all wavelengths.

        Args:
            ct_matrix: 3D crosstalk matrix (n_pixels x n_pixels x n_wl).
            wavelengths: Wavelength array in um.

        Returns:
            Tuple of (peak_crosstalk_value, wavelength_um).
        """
        spectral_ct = CrosstalkAnalyzer.spectral_crosstalk(ct_matrix, wavelengths)
        if len(spectral_ct) == 0:
            return 0.0, 0.0
        idx = int(np.argmax(spectral_ct))
        return float(spectral_ct[idx]), float(wavelengths[idx])

    @staticmethod
    def neighbor_crosstalk(
        ct_matrix: np.ndarray,
        bayer_map: list,
    ) -> Dict[str, float]:
        """Nearest-neighbour crosstalk for each pixel.

        Uses the Bayer map to determine pixel adjacency.  For each pixel
        the average crosstalk from its nearest neighbours is returned,
        averaged over all wavelengths.

        Args:
            ct_matrix: 3D crosstalk matrix (n_pixels x n_pixels x n_wl).
            bayer_map: Bayer pattern as a 2D list (e.g. [["R","G"],["G","B"]]).

        Returns:
            Dictionary mapping pixel name (e.g. "R_0_0") to average
            nearest-neighbour crosstalk.
        """
        # Build flat pixel list and row/col lookup
        rows = len(bayer_map)
        cols = len(bayer_map[0]) if rows > 0 else 0

        pixel_names: List[str] = []
        pos_map: Dict[str, Tuple[int, int]] = {}
        for r in range(rows):
            for c in range(cols):
                name = f"{bayer_map[r][c]}_{r}_{c}"
                pixel_names.append(name)
                pos_map[name] = (r, c)

        pixel_names.sort()
        n_pixels = len(pixel_names)
        name_to_idx = {name: idx for idx, name in enumerate(pixel_names)}

        neighbor_ct: Dict[str, float] = {}

        for name in pixel_names:
            r, c = pos_map[name]
            i = name_to_idx[name]
            # 4-connected neighbours
            neighbours = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nname = f"{bayer_map[nr][nc]}_{nr}_{nc}"
                    if nname in name_to_idx:
                        neighbours.append(name_to_idx[nname])

            if not neighbours:
                neighbor_ct[name] = 0.0
                continue

            # Average crosstalk from neighbours, averaged over wavelengths
            ct_vals = []
            for j in neighbours:
                ct_vals.append(float(np.mean(ct_matrix[i, j, :])))
            neighbor_ct[name] = float(np.mean(ct_vals))

        return neighbor_ct

    @staticmethod
    def summarize(result: SimulationResult) -> dict:
        """Produce a dictionary of key crosstalk metrics.

        This is a convenience entry-point that builds the crosstalk matrix
        from a SimulationResult and computes all summary statistics.

        Args:
            result: A SimulationResult containing qe_per_pixel and
                wavelengths.

        Returns:
            Dictionary with keys:
                - "n_pixels": number of pixels
                - "n_wavelengths": number of wavelengths
                - "peak_crosstalk": maximum average off-diagonal crosstalk
                - "peak_wavelength_um": wavelength at which peak occurs
                - "mean_crosstalk": mean off-diagonal crosstalk over spectrum
                - "spectral_crosstalk": array of per-wavelength avg crosstalk
        """
        if not result.qe_per_pixel:
            logger.warning("No QE data available for crosstalk summary.")
            return {
                "n_pixels": 0,
                "n_wavelengths": len(result.wavelengths),
                "peak_crosstalk": 0.0,
                "peak_wavelength_um": 0.0,
                "mean_crosstalk": 0.0,
                "spectral_crosstalk": np.array([]),
            }

        pixels = sorted(result.qe_per_pixel.keys())
        bayer_map_flat = pixels  # used only for ordering in compute_matrix

        ct_matrix = CrosstalkAnalyzer.compute_matrix(
            result.qe_per_pixel, bayer_map_flat, result.wavelengths,
        )
        spectral_ct = CrosstalkAnalyzer.spectral_crosstalk(
            ct_matrix, result.wavelengths,
        )
        peak_val, peak_wl = CrosstalkAnalyzer.peak_crosstalk(
            ct_matrix, result.wavelengths,
        )

        return {
            "n_pixels": len(pixels),
            "n_wavelengths": len(result.wavelengths),
            "peak_crosstalk": peak_val,
            "peak_wavelength_um": peak_wl,
            "mean_crosstalk": float(np.mean(spectral_ct)) if len(spectral_ct) > 0 else 0.0,
            "spectral_crosstalk": spectral_ct,
        }
