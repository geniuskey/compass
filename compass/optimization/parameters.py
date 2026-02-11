"""Optimizable parameter definitions for COMPASS inverse design.

Each parameter class wraps a specific physical quantity in the pixel
configuration (e.g. microlens height, BARL thicknesses) and provides
a uniform interface for the optimizer to read/write values, query
bounds, and convert to/from flat numpy vectors.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class OptimizableParameter(ABC):
    """Base class for parameters that can be optimized.

    Subclasses must implement get/set on the config dict, declare bounds,
    and provide a human-readable name and parameter size.
    """

    @abstractmethod
    def get_value(self) -> np.ndarray:
        """Return current parameter value(s) as a 1-D array."""

    @abstractmethod
    def set_value(self, value: np.ndarray) -> None:
        """Set parameter value(s) from a 1-D array."""

    @abstractmethod
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower_bounds, upper_bounds) arrays."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable parameter name."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of scalar parameters."""


class MicrolensHeight(OptimizableParameter):
    """Optimize microlens sag height (um).

    Modifies ``config["pixel"]["layers"]["microlens"]["height"]``.

    Args:
        config: Full COMPASS config dict (modified in place).
        min_val: Lower bound in um.
        max_val: Upper bound in um.
    """

    def __init__(self, config: dict, min_val: float = 0.1, max_val: float = 1.5):
        self.config = config
        self.min_val = min_val
        self.max_val = max_val

    def _ml_cfg(self) -> dict:
        return self.config.setdefault("pixel", {}).setdefault("layers", {}).setdefault(
            "microlens", {}
        )

    def get_value(self) -> np.ndarray:
        return np.array([self._ml_cfg().get("height", 0.6)])

    def set_value(self, value: np.ndarray) -> None:
        self._ml_cfg()["height"] = float(np.clip(value[0], self.min_val, self.max_val))

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self.min_val]), np.array([self.max_val])

    @property
    def name(self) -> str:
        return "microlens_height"

    @property
    def size(self) -> int:
        return 1


class MicrolensSquareness(OptimizableParameter):
    """Optimize superellipse squareness parameter n.

    Modifies ``config["pixel"]["layers"]["microlens"]["profile"]["n"]``.

    Args:
        config: Full COMPASS config dict.
        min_val: Lower bound for n.
        max_val: Upper bound for n.
    """

    def __init__(self, config: dict, min_val: float = 1.5, max_val: float = 5.0):
        self.config = config
        self.min_val = min_val
        self.max_val = max_val

    def _profile_cfg(self) -> dict:
        return (
            self.config.setdefault("pixel", {})
            .setdefault("layers", {})
            .setdefault("microlens", {})
            .setdefault("profile", {})
        )

    def get_value(self) -> np.ndarray:
        return np.array([self._profile_cfg().get("n", 2.5)])

    def set_value(self, value: np.ndarray) -> None:
        self._profile_cfg()["n"] = float(np.clip(value[0], self.min_val, self.max_val))

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self.min_val]), np.array([self.max_val])

    @property
    def name(self) -> str:
        return "microlens_squareness"

    @property
    def size(self) -> int:
        return 1


class BARLThicknesses(OptimizableParameter):
    """Optimize BARL (Bottom Anti-Reflective Layer) thicknesses jointly.

    Modifies each element of
    ``config["pixel"]["layers"]["barl"]["layers"][i]["thickness"]``.

    Args:
        config: Full COMPASS config dict.
        min_val: Lower bound per layer in um.
        max_val: Upper bound per layer in um.
    """

    def __init__(self, config: dict, min_val: float = 0.005, max_val: float = 0.1):
        self.config = config
        self.min_val = min_val
        self.max_val = max_val

    def _barl_layers(self) -> list[dict]:
        return (
            self.config.setdefault("pixel", {})
            .setdefault("layers", {})
            .setdefault("barl", {})
            .setdefault("layers", [])
        )

    def get_value(self) -> np.ndarray:
        layers = self._barl_layers()
        if not layers:
            return np.array([])
        return np.array([bl.get("thickness", 0.01) for bl in layers])

    def set_value(self, value: np.ndarray) -> None:
        layers = self._barl_layers()
        for i, bl in enumerate(layers):
            if i < len(value):
                bl["thickness"] = float(np.clip(value[i], self.min_val, self.max_val))

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._barl_layers())
        return np.full(n, self.min_val), np.full(n, self.max_val)

    @property
    def name(self) -> str:
        return "barl_thicknesses"

    @property
    def size(self) -> int:
        return len(self._barl_layers())


class ColorFilterThickness(OptimizableParameter):
    """Optimize color filter thickness (um).

    Modifies ``config["pixel"]["layers"]["color_filter"]["thickness"]``.

    Args:
        config: Full COMPASS config dict.
        min_val: Lower bound in um.
        max_val: Upper bound in um.
    """

    def __init__(self, config: dict, min_val: float = 0.3, max_val: float = 1.2):
        self.config = config
        self.min_val = min_val
        self.max_val = max_val

    def _cf_cfg(self) -> dict:
        return (
            self.config.setdefault("pixel", {})
            .setdefault("layers", {})
            .setdefault("color_filter", {})
        )

    def get_value(self) -> np.ndarray:
        return np.array([self._cf_cfg().get("thickness", 0.6)])

    def set_value(self, value: np.ndarray) -> None:
        self._cf_cfg()["thickness"] = float(
            np.clip(value[0], self.min_val, self.max_val)
        )

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self.min_val]), np.array([self.max_val])

    @property
    def name(self) -> str:
        return "color_filter_thickness"

    @property
    def size(self) -> int:
        return 1


class MicrolensRadii(OptimizableParameter):
    """Optimize microlens radii (radius_x, radius_y) in um.

    Modifies ``config["pixel"]["layers"]["microlens"]["radius_x"]``
    and ``config["pixel"]["layers"]["microlens"]["radius_y"]``.

    Args:
        config: Full COMPASS config dict.
        min_val: Lower bound for both radii.
        max_val: Upper bound for both radii.
    """

    def __init__(self, config: dict, min_val: float = 0.2, max_val: float = 0.6):
        self.config = config
        self.min_val = min_val
        self.max_val = max_val

    def _ml_cfg(self) -> dict:
        return self.config.setdefault("pixel", {}).setdefault("layers", {}).setdefault(
            "microlens", {}
        )

    def get_value(self) -> np.ndarray:
        cfg = self._ml_cfg()
        return np.array([cfg.get("radius_x", 0.48), cfg.get("radius_y", 0.48)])

    def set_value(self, value: np.ndarray) -> None:
        cfg = self._ml_cfg()
        cfg["radius_x"] = float(np.clip(value[0], self.min_val, self.max_val))
        cfg["radius_y"] = float(np.clip(value[1], self.min_val, self.max_val))

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.full(2, self.min_val), np.full(2, self.max_val)

    @property
    def name(self) -> str:
        return "microlens_radii"

    @property
    def size(self) -> int:
        return 2


class ParameterSpace:
    """Collection of optimizable parameters with flat-vector conversion.

    Manages converting between a single flat numpy vector (for the optimizer)
    and the individual parameter objects that modify the config dict.

    Args:
        params: List of OptimizableParameter instances.
    """

    def __init__(self, params: list[OptimizableParameter]):
        if not params:
            raise ValueError("ParameterSpace requires at least one parameter.")
        self.params = params

    @property
    def total_size(self) -> int:
        """Total number of scalar parameters across all optimizables."""
        return sum(p.size for p in self.params)

    @property
    def names(self) -> list[str]:
        """Ordered list of parameter names."""
        return [p.name for p in self.params]

    def to_vector(self) -> np.ndarray:
        """Read current values from all parameters into a flat vector."""
        parts = [p.get_value() for p in self.params]
        if not parts:
            return np.array([])
        return np.concatenate(parts)

    def from_vector(self, x: np.ndarray) -> None:
        """Distribute a flat vector back into individual parameters.

        Args:
            x: Flat array of length ``total_size``.
        """
        if len(x) != self.total_size:
            raise ValueError(
                f"Vector length {len(x)} != total parameter size {self.total_size}"
            )
        offset = 0
        for p in self.params:
            p.set_value(x[offset : offset + p.size])
            offset += p.size

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Concatenated (lower, upper) bounds for the full vector."""
        lowers = []
        uppers = []
        for p in self.params:
            lo, hi = p.get_bounds()
            lowers.append(lo)
            uppers.append(hi)
        return np.concatenate(lowers), np.concatenate(uppers)

    def describe(self) -> list[dict]:
        """Return a description of each parameter for reporting."""
        descriptions = []
        for p in self.params:
            lo, hi = p.get_bounds()
            descriptions.append({
                "name": p.name,
                "size": p.size,
                "current_value": p.get_value().tolist(),
                "lower_bound": lo.tolist(),
                "upper_bound": hi.tolist(),
            })
        return descriptions
