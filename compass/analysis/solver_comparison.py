"""Solver comparison analysis module."""
from __future__ import annotations

import numpy as np

from compass.core.types import SimulationResult


class SolverComparison:
    """Compare results from multiple solvers."""

    #: Relative errors are taken against max(|QE_ref|, this x peak |QE_ref|)
    #: so near-zero out-of-band QE does not dominate the maximum.
    RELATIVE_FLOOR = 0.01

    def __init__(self, results: list[SimulationResult], labels: list[str], reference_idx: int = 0):
        if len(results) != len(labels):
            raise ValueError(f"{len(results)} results but {len(labels)} labels")
        ref_wl = np.asarray(results[reference_idx].wavelengths)
        for result, label in zip(results, labels, strict=True):
            wl = np.asarray(result.wavelengths)
            if wl.shape != ref_wl.shape or not np.allclose(wl, ref_wl, rtol=0, atol=1e-9):
                raise ValueError(
                    f"solver {label!r} was run on a different wavelength grid than the "
                    "reference; comparisons are element-wise"
                )
        self.results = results
        self.labels = labels
        self.reference_idx = reference_idx

    def qe_difference(self) -> dict[str, np.ndarray]:
        """Absolute QE difference vs reference for each pixel."""
        ref = self.results[self.reference_idx]
        diffs = {}
        for i, (result, label) in enumerate(zip(self.results, self.labels)):
            if i == self.reference_idx:
                continue
            for pixel in ref.qe_per_pixel:
                if pixel in result.qe_per_pixel:
                    key = f"{label}_vs_{self.labels[self.reference_idx]}_{pixel}"
                    diffs[key] = np.abs(result.qe_per_pixel[pixel] - ref.qe_per_pixel[pixel])
        return diffs

    def qe_relative_error(self) -> dict[str, np.ndarray]:
        """Relative QE error (%) vs reference.

        The denominator is floored at ``RELATIVE_FLOOR`` times the pixel's
        peak reference QE, so out-of-band wavelengths where QE ~ 0 (e.g. a
        blue pixel at 700 nm) do not produce meaningless 1e8 % errors.
        """
        ref = self.results[self.reference_idx]
        errors = {}
        for i, (result, label) in enumerate(zip(self.results, self.labels)):
            if i == self.reference_idx:
                continue
            for pixel in ref.qe_per_pixel:
                if pixel in result.qe_per_pixel:
                    key = f"{label}_vs_{self.labels[self.reference_idx]}_{pixel}"
                    ref_qe = ref.qe_per_pixel[pixel]
                    ref_abs = np.abs(np.asarray(ref_qe, dtype=float))
                    peak = float(np.nanmax(ref_abs)) if ref_abs.size else 0.0
                    floor = max(1e-10, self.RELATIVE_FLOOR * peak)
                    errors[key] = (
                        100.0 * np.abs(result.qe_per_pixel[pixel] - ref_qe)
                        / np.maximum(ref_abs, floor)
                    )
        return errors

    def runtime_comparison(self) -> dict[str, float]:
        """Compare runtime across solvers."""
        return {label: result.metadata.get("runtime_seconds", 0.0) for label, result in zip(self.labels, self.results)}

    def summary(self) -> dict:
        """Generate comparison summary."""
        qe_diff = self.qe_difference()
        qe_rel = self.qe_relative_error()
        runtimes = self.runtime_comparison()
        return {
            "max_qe_diff": {k: float(np.max(v)) for k, v in qe_diff.items()},
            "mean_qe_diff": {k: float(np.mean(v)) for k, v in qe_diff.items()},
            "max_qe_relative_error_pct": {k: float(np.max(v)) for k, v in qe_rel.items()},
            "runtimes_seconds": runtimes,
        }
