"""Solver comparison analysis module."""
from __future__ import annotations

import numpy as np

from compass.core.types import SimulationResult


class SolverComparison:
    """Compare results from multiple solvers."""

    def __init__(self, results: list[SimulationResult], labels: list[str], reference_idx: int = 0):
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
        """Relative QE error (%) vs reference."""
        ref = self.results[self.reference_idx]
        errors = {}
        for i, (result, label) in enumerate(zip(self.results, self.labels)):
            if i == self.reference_idx:
                continue
            for pixel in ref.qe_per_pixel:
                if pixel in result.qe_per_pixel:
                    key = f"{label}_vs_{self.labels[self.reference_idx]}_{pixel}"
                    ref_qe = ref.qe_per_pixel[pixel]
                    errors[key] = 100.0 * np.abs(result.qe_per_pixel[pixel] - ref_qe) / np.maximum(np.abs(ref_qe), 1e-10)
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
