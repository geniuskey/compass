"""Dedicated crosstalk analysis module for COMPASS.

A single full-illumination simulation cannot tag photons by the pixel they
were "intended" for, so crosstalk is defined spectrally (consistent with
:meth:`compass.analysis.qe_calculator.QECalculator.compute_crosstalk`):
at wavelength λ, the intended pixels are those whose color channel band
contains λ, and crosstalk is the share of collected signal that lands in
pixels of a *different* color.
"""

from __future__ import annotations

import logging

import numpy as np

from compass.analysis.qe_calculator import QECalculator
from compass.core.types import SimulationResult

logger = logging.getLogger(__name__)


def _pixel_color(name: str) -> str:
    """Return the color prefix of a pixel name such as ``"R_0_0"``."""
    return name.split("_")[0]


class CrosstalkAnalyzer:
    """Analyse pixel-to-pixel spectral crosstalk from QE simulation data.

    This class builds a wavelength-resolved crosstalk matrix and extracts
    summary statistics suitable for automated reporting.
    """

    @staticmethod
    def compute_matrix(
        qe_per_pixel: dict[str, np.ndarray],
        bayer_map: list | None,
        wavelengths: np.ndarray,
        bands: dict[str, tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Compute the 3D spectral crosstalk matrix.

        ``ct[i, j, k]`` is the fraction of the total collected signal at
        wavelength ``k`` that lands in pixel ``j``, defined only where
        wavelength ``k`` lies inside pixel ``i``'s color band (i.e. where
        light at that wavelength is intended for pixel ``i``). Entries where
        the wavelength is outside pixel ``i``'s band are ``NaN``. Defined
        rows sum to 1. Pixels with an unknown color prefix use the full
        simulated range as their band.

        Args:
            qe_per_pixel: Mapping of pixel name to QE spectrum array.
                Pixels are ordered by sorted name.
            bayer_map: Unused; kept for API compatibility.
            wavelengths: Wavelength array in um.
            bands: Optional override of per-channel wavelength bands (um).
                Defaults to :attr:`QECalculator.CHANNEL_BANDS`.

        Returns:
            3D numpy array of shape (n_pixels, n_pixels, n_wavelengths).
        """
        del bayer_map
        wavelengths = np.asarray(wavelengths, dtype=float)
        bands = bands if bands is not None else QECalculator.CHANNEL_BANDS
        pixels = sorted(qe_per_pixel.keys())
        n_pixels = len(pixels)
        n_wl = len(wavelengths)
        ct = np.full((n_pixels, n_pixels, n_wl), np.nan)
        if n_pixels == 0 or n_wl == 0:
            return ct

        qe = np.stack([np.asarray(qe_per_pixel[p], dtype=float) for p in pixels])
        if qe.shape[1] != n_wl:
            raise ValueError(
                f"QE spectra have {qe.shape[1]} samples but {n_wl} wavelengths were given"
            )
        total = qe.sum(axis=0)
        share = qe / np.where(total > 0, total, 1.0)

        full_range = (float(wavelengths.min()), float(wavelengths.max()))
        for i, pi in enumerate(pixels):
            lo, hi = bands.get(_pixel_color(pi), full_range)
            in_band = (wavelengths >= lo) & (wavelengths <= hi)
            for k in np.flatnonzero(in_band):
                if total[k] > 0:
                    ct[i, :, k] = share[:, k]
                else:
                    ct[i, :, k] = 0.0
                    ct[i, i, k] = 1.0
        return ct

    @staticmethod
    def spectral_crosstalk(
        ct_matrix: np.ndarray,
        wavelengths: np.ndarray,
        pixel_names: list[str] | None = None,
    ) -> np.ndarray:
        """Per-wavelength crosstalk.

        For each wavelength, the crosstalk of each intended (defined) row is
        the summed share collected by pixels of a different color; the
        result is the average over intended rows. Without ``pixel_names``,
        every off-diagonal pixel counts as crosstalk.

        Args:
            ct_matrix: 3D crosstalk matrix (n_pixels x n_pixels x n_wl).
            wavelengths: Wavelength array in um (length n_wl).
            pixel_names: Sorted pixel names matching the matrix ordering,
                used to treat same-color pixels as correct collection.

        Returns:
            1D array of length n_wl. ``NaN`` where no pixel's band contains
            the wavelength.
        """
        del wavelengths
        n_pixels, _, n_wl = ct_matrix.shape
        if pixel_names is not None and len(pixel_names) != n_pixels:
            raise ValueError(
                f"pixel_names has {len(pixel_names)} entries but matrix has {n_pixels} pixels"
            )
        if pixel_names is not None:
            colors = [_pixel_color(p) for p in pixel_names]
            other = np.array([[ci != cj for cj in colors] for ci in colors], dtype=float)
        else:
            other = 1.0 - np.eye(n_pixels)

        avg_ct = np.full(n_wl, np.nan)
        for k in range(n_wl):
            mat = ct_matrix[:, :, k]
            defined = ~np.isnan(mat[:, 0]) if n_pixels else np.zeros(0, dtype=bool)
            if not np.any(defined):
                continue
            per_row = np.sum(mat[defined] * other[defined], axis=1)
            avg_ct[k] = float(np.mean(per_row))
        return avg_ct

    @staticmethod
    def peak_crosstalk(
        ct_matrix: np.ndarray,
        wavelengths: np.ndarray,
        pixel_names: list[str] | None = None,
    ) -> tuple[float, float]:
        """Find peak crosstalk value and corresponding wavelength.

        Args:
            ct_matrix: 3D crosstalk matrix (n_pixels x n_pixels x n_wl).
            wavelengths: Wavelength array in um.
            pixel_names: Optional sorted pixel names (see
                :meth:`spectral_crosstalk`).

        Returns:
            Tuple of (peak_crosstalk_value, wavelength_um). ``(nan, nan)``
            when crosstalk is undefined at every wavelength; ``(0, 0)`` for
            an empty spectrum.
        """
        spectral_ct = CrosstalkAnalyzer.spectral_crosstalk(ct_matrix, wavelengths, pixel_names)
        if len(spectral_ct) == 0:
            return 0.0, 0.0
        if np.all(np.isnan(spectral_ct)):
            return float("nan"), float("nan")
        idx = int(np.nanargmax(spectral_ct))
        return float(spectral_ct[idx]), float(np.asarray(wavelengths)[idx])

    @staticmethod
    def neighbor_crosstalk(
        ct_matrix: np.ndarray,
        bayer_map: list,
    ) -> dict[str, float]:
        """Nearest-neighbour crosstalk for each pixel.

        Uses the Bayer map to determine 4-connected pixel adjacency (no
        periodic wrap). For each pixel, the crosstalk into each neighbour is
        averaged over the wavelengths where that pixel is intended, then
        averaged over neighbours.

        Args:
            ct_matrix: 3D crosstalk matrix (n_pixels x n_pixels x n_wl),
                ordered by sorted pixel name.
            bayer_map: Bayer pattern as a 2D list (e.g. [["R","G"],["G","B"]]).

        Returns:
            Dictionary mapping pixel name (e.g. "R_0_0") to average
            nearest-neighbour crosstalk (``NaN`` if never intended).
        """
        rows = len(bayer_map)
        cols = len(bayer_map[0]) if rows > 0 else 0

        pos_map: dict[str, tuple[int, int]] = {
            f"{bayer_map[r][c]}_{r}_{c}": (r, c) for r in range(rows) for c in range(cols)
        }
        pixel_names = sorted(pos_map)
        if len(pixel_names) != ct_matrix.shape[0]:
            raise ValueError(
                f"bayer_map defines {len(pixel_names)} pixels but matrix has "
                f"{ct_matrix.shape[0]}"
            )
        name_to_idx = {name: idx for idx, name in enumerate(pixel_names)}

        neighbor_ct: dict[str, float] = {}
        for name in pixel_names:
            r, c = pos_map[name]
            i = name_to_idx[name]
            neighbours = [
                name_to_idx[f"{bayer_map[r + dr][c + dc]}_{r + dr}_{c + dc}"]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= r + dr < rows and 0 <= c + dc < cols
            ]
            if not neighbours:
                neighbor_ct[name] = 0.0
                continue
            row = ct_matrix[i, neighbours, :]
            if np.all(np.isnan(row)):
                neighbor_ct[name] = float("nan")
                continue
            neighbor_ct[name] = float(np.mean(np.nanmean(row, axis=1)))
        return neighbor_ct

    @staticmethod
    def summarize(result: SimulationResult) -> dict:
        """Produce a dictionary of key crosstalk metrics.

        Args:
            result: A SimulationResult containing qe_per_pixel and
                wavelengths.

        Returns:
            Dictionary with keys:
                - "n_pixels": number of pixels
                - "n_wavelengths": number of wavelengths
                - "peak_crosstalk": maximum per-wavelength crosstalk
                - "peak_wavelength_um": wavelength at which peak occurs
                - "mean_crosstalk": mean crosstalk over wavelengths where
                  it is defined
                - "spectral_crosstalk": array of per-wavelength crosstalk
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
        ct_matrix = CrosstalkAnalyzer.compute_matrix(
            result.qe_per_pixel, None, result.wavelengths,
        )
        spectral_ct = CrosstalkAnalyzer.spectral_crosstalk(
            ct_matrix, result.wavelengths, pixels,
        )
        peak_val, peak_wl = CrosstalkAnalyzer.peak_crosstalk(
            ct_matrix, result.wavelengths, pixels,
        )
        defined = spectral_ct[~np.isnan(spectral_ct)]

        return {
            "n_pixels": len(pixels),
            "n_wavelengths": len(result.wavelengths),
            "peak_crosstalk": peak_val,
            "peak_wavelength_um": peak_wl,
            "mean_crosstalk": float(np.mean(defined)) if len(defined) > 0 else float("nan"),
            "spectral_crosstalk": spectral_ct,
        }
