"""Planewave source model for COMPASS simulations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

from compass.core.units import deg_to_rad


@dataclass
class PlanewaveSource:
    """Planewave optical source definition.

    Supports single wavelength, sweep, and list modes.
    Handles TE/TM/unpolarized decomposition.

    Attributes:
        wavelengths: Array of wavelengths in um.
        theta_deg: Polar incidence angle in degrees.
        phi_deg: Azimuthal incidence angle in degrees.
        polarization: "TE", "TM", or "unpolarized".
    """

    wavelengths: np.ndarray
    theta_deg: float = 0.0
    phi_deg: float = 0.0
    polarization: Literal["TE", "TM", "unpolarized"] = "unpolarized"

    def __post_init__(self) -> None:
        if self.polarization not in ("TE", "TM", "unpolarized"):
            raise ValueError(
                f"polarization must be 'TE', 'TM' or 'unpolarized', got {self.polarization!r}"
            )
        if len(self.wavelengths) == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        if np.any(self.wavelengths < 0.1) or np.any(self.wavelengths > 100):
            warnings.warn(
                f"PlanewaveSource: wavelengths outside typical range [0.1, 100] um "
                f"(min={float(np.min(self.wavelengths)):.4f}, "
                f"max={float(np.max(self.wavelengths)):.4f})",
                stacklevel=2,
            )

    @classmethod
    def from_config(cls, source_config: dict) -> PlanewaveSource:
        """Create PlanewaveSource from source config dictionary.

        Args:
            source_config: Source configuration with wavelength, angle, polarization.

        Returns:
            PlanewaveSource instance.
        """
        # `or {}`: schema-dumped configs carry present-but-None sections.
        wl_cfg = source_config.get("wavelength") or {}
        mode = wl_cfg.get("mode") or "single"

        if mode == "single":
            value = wl_cfg.get("value")
            wavelengths = np.array([0.55 if value is None else value], dtype=float)
        elif mode == "sweep":
            sweep = wl_cfg.get("sweep") or {}
            start = sweep.get("start", 0.38)
            stop = sweep.get("stop", 0.78)
            step = sweep.get("step", 0.01)
            if step is None or step <= 0:
                raise ValueError(f"wavelength sweep step must be positive, got {step}")
            wavelengths = np.arange(start, stop + step / 2, step)
        elif mode == "list":
            values = wl_cfg.get("values")
            if not values:
                raise ValueError("wavelengths array must not be empty (mode 'list' needs 'values')")
            wavelengths = np.atleast_1d(np.asarray(values, dtype=float))
        else:
            raise ValueError(f"Unknown wavelength mode: {mode}")

        angle_cfg = source_config.get("angle") or {}

        return cls(
            wavelengths=wavelengths,
            theta_deg=angle_cfg.get("theta_deg", 0.0),
            phi_deg=angle_cfg.get("phi_deg", 0.0),
            polarization=source_config.get("polarization") or "unpolarized",
        )

    @property
    def theta_rad(self) -> float:
        """Polar angle in radians."""
        return deg_to_rad(self.theta_deg)

    @property
    def phi_rad(self) -> float:
        """Azimuthal angle in radians."""
        return deg_to_rad(self.phi_deg)

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelengths."""
        return len(self.wavelengths)

    @property
    def is_unpolarized(self) -> bool:
        """Whether simulation needs TE+TM averaging."""
        return self.polarization == "unpolarized"

    def get_polarization_runs(self) -> list[str]:
        """Get list of polarization states to simulate.

        For unpolarized, returns ["TE", "TM"] for averaging.
        """
        if self.is_unpolarized:
            return ["TE", "TM"]
        return [self.polarization]

    def to_solver_params(self) -> dict:
        """Convert to solver-compatible parameter dictionary."""
        return {
            "wavelengths": self.wavelengths,
            "theta_rad": self.theta_rad,
            "phi_rad": self.phi_rad,
            "polarization": self.polarization,
            "polarization_runs": self.get_polarization_runs(),
        }
