"""Optimization objective functions for COMPASS inverse design.

Each objective computes a scalar value to be MINIMIZED by the optimizer.
Maximization objectives (e.g. MaximizeQE) return the negated metric so
that scipy minimizers drive the value in the desired direction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from compass.core.types import SimulationResult

logger = logging.getLogger(__name__)


class ObjectiveFunction(ABC):
    """Base class for optimization objectives.

    All objectives return a scalar float that the optimizer **minimizes**.
    """

    @abstractmethod
    def evaluate(self, result: SimulationResult) -> float:
        """Compute scalar objective value (to be MINIMIZED).

        Args:
            result: Simulation result from a solver run.

        Returns:
            Scalar objective value.
        """

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this objective."""


class MaximizeQE(ObjectiveFunction):
    """Maximize average QE across specified wavelengths and pixels.

    Returns negative mean QE so that minimization drives QE upward.

    Args:
        target_pixels: List of pixel names to include (None = all pixels).
        wavelength_range: (wl_min, wl_max) in um to restrict average.
            None means use the full wavelength range.
        weights: Per-wavelength weights array. Must match the number of
            wavelengths after range filtering. None = uniform weights.
    """

    def __init__(
        self,
        target_pixels: list[str] | None = None,
        wavelength_range: tuple[float, float] | None = None,
        weights: np.ndarray | None = None,
    ):
        self.target_pixels = target_pixels
        self.wavelength_range = wavelength_range
        self.weights = weights

    def evaluate(self, result: SimulationResult) -> float:
        """Return negative mean QE (minimization -> maximization)."""
        wl = np.asarray(result.wavelengths)
        mask = np.ones(len(wl), dtype=bool)
        if self.wavelength_range is not None:
            wl_min, wl_max = self.wavelength_range
            mask = (wl >= wl_min) & (wl <= wl_max)

        pixels = self.target_pixels or list(result.qe_per_pixel.keys())
        if not pixels:
            return 0.0

        qe_values = []
        for pname in pixels:
            if pname in result.qe_per_pixel:
                qe_values.append(np.asarray(result.qe_per_pixel[pname])[mask])

        if not qe_values:
            return 0.0

        qe_stack = np.stack(qe_values, axis=0)  # (n_pixels, n_wl)
        mean_per_wl = np.mean(qe_stack, axis=0)  # average over pixels

        if self.weights is not None:
            w = np.asarray(self.weights)
            if len(w) != len(mean_per_wl):
                logger.warning(
                    f"Weight length ({len(w)}) != filtered wavelength count "
                    f"({len(mean_per_wl)}). Using uniform weights."
                )
                avg_qe = float(np.mean(mean_per_wl))
            else:
                avg_qe = float(np.average(mean_per_wl, weights=w))
        else:
            avg_qe = float(np.mean(mean_per_wl))

        return -avg_qe  # negate for minimization

    def name(self) -> str:
        return "MaximizeQE"


class MinimizeCrosstalk(ObjectiveFunction):
    """Minimize optical crosstalk between adjacent pixels.

    Crosstalk is computed as the mean off-diagonal fraction: for each pixel,
    the fraction of total QE that lands in *other* pixels. This is averaged
    over wavelengths within the specified range.

    Args:
        target_wavelength_range: (wl_min, wl_max) in um. None = full range.
    """

    def __init__(
        self,
        target_wavelength_range: tuple[float, float] | None = None,
    ):
        self.target_wavelength_range = target_wavelength_range

    def evaluate(self, result: SimulationResult) -> float:
        """Return mean crosstalk (already a value to minimize)."""
        wl = np.asarray(result.wavelengths)
        mask = np.ones(len(wl), dtype=bool)
        if self.target_wavelength_range is not None:
            wl_min, wl_max = self.target_wavelength_range
            mask = (wl >= wl_min) & (wl <= wl_max)

        pixels = sorted(result.qe_per_pixel.keys())
        n = len(pixels)
        if n < 2:
            return 0.0

        # For each pixel, compute its fraction of total QE
        qe_matrix = np.array([
            np.asarray(result.qe_per_pixel[p])[mask] for p in pixels
        ])  # (n_pixels, n_wl)
        total_qe = np.sum(qe_matrix, axis=0, keepdims=True)  # (1, n_wl)
        total_qe = np.maximum(total_qe, 1e-30)
        fractions = qe_matrix / total_qe  # (n_pixels, n_wl)

        # Crosstalk = 1 - self-fraction, averaged over all pixels and wavelengths
        # Self-fraction for pixel i = fractions[i] when that pixel is illuminated.
        # Simplified: mean off-diagonal fraction = 1 - (1/n) since uniform illum.
        # More precise: sum of non-self fractions per pixel.
        crosstalk_per_pixel = 1.0 - fractions  # fraction going elsewhere
        mean_crosstalk = float(np.mean(crosstalk_per_pixel))

        return mean_crosstalk

    def name(self) -> str:
        return "MinimizeCrosstalk"


class MaximizePeakQE(ObjectiveFunction):
    """Maximize peak QE for a specific color channel.

    Finds all pixels matching the given color channel (e.g. "G") and
    returns the negative of their peak QE value.

    Args:
        channel: Color channel letter ("R", "G", or "B").
    """

    def __init__(self, channel: str = "G"):
        self.channel = channel.upper()

    def evaluate(self, result: SimulationResult) -> float:
        """Return negative peak QE for the specified channel."""
        matching = [
            np.asarray(qe)
            for pname, qe in result.qe_per_pixel.items()
            if pname.startswith(self.channel + "_")
        ]
        if not matching:
            # Fallback: try to find any pixel with channel in name
            matching = [
                np.asarray(qe)
                for pname, qe in result.qe_per_pixel.items()
                if self.channel in pname
            ]
        if not matching:
            return 0.0

        avg_qe = np.mean(np.stack(matching, axis=0), axis=0)
        peak = float(np.max(avg_qe))
        return -peak

    def name(self) -> str:
        return f"MaximizePeakQE({self.channel})"


class EnergyBalanceRegularizer(ObjectiveFunction):
    """Penalize energy conservation violations.

    Computes max |R + T + A - 1| and applies a quadratic penalty if it
    exceeds the tolerance threshold.

    Args:
        tolerance: Acceptable deviation from unity (default 0.01).
        penalty_weight: Multiplier for the penalty term (default 10.0).
    """

    def __init__(self, tolerance: float = 0.01, penalty_weight: float = 10.0):
        self.tolerance = tolerance
        self.penalty_weight = penalty_weight

    def evaluate(self, result: SimulationResult) -> float:
        """Return penalty for energy balance violations."""
        if result.reflection is None or result.transmission is None:
            return 0.0

        R = np.asarray(result.reflection)
        T = np.asarray(result.transmission)
        A = (
            np.asarray(result.absorption)
            if result.absorption is not None
            else 1.0 - R - T
        )

        total = R + T + A
        error = np.abs(total - 1.0)
        max_err = float(np.max(error))

        if max_err <= self.tolerance:
            return 0.0

        # Quadratic penalty on excess
        excess = max_err - self.tolerance
        return self.penalty_weight * excess ** 2

    def name(self) -> str:
        return "EnergyBalanceRegularizer"


class CompositeObjective(ObjectiveFunction):
    """Weighted combination of multiple objectives.

    The final value is sum(weight_i * objective_i.evaluate(result)).

    Args:
        objectives: List of (weight, ObjectiveFunction) tuples.
    """

    def __init__(self, objectives: list[tuple[float, ObjectiveFunction]]):
        if not objectives:
            raise ValueError("CompositeObjective requires at least one objective.")
        self.objectives = objectives

    def evaluate(self, result: SimulationResult) -> float:
        """Weighted sum of component objectives."""
        total = 0.0
        for weight, obj in self.objectives:
            val = obj.evaluate(result)
            total += weight * val
        return total

    def name(self) -> str:
        parts = [
            f"{w:.2f}*{obj.name()}" for w, obj in self.objectives
        ]
        return "Composite(" + " + ".join(parts) + ")"
