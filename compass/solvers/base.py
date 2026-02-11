"""Abstract base class for all EM solvers in COMPASS."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack

logger = logging.getLogger(__name__)


class SolverBase(ABC):
    """Abstract base class for all EM solvers.

    All solver adapters (RCWA, FDTD) must implement this interface
    to ensure uniform access from runners and analysis modules.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """Initialize solver.

        Args:
            config: Solver configuration dictionary (from Hydra).
            device: Compute device ("cpu", "cuda", "mps").
        """
        self.config = config
        self.device = device
        self._pixel_stack: PixelStack | None = None
        self._source_config: dict | None = None

    @property
    def name(self) -> str:
        """Solver name."""
        return str(self.config.get("name", self.__class__.__name__))

    @property
    def solver_type(self) -> str:
        """Solver type (rcwa or fdtd)."""
        return str(self.config.get("type", "unknown"))

    @abstractmethod
    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        """Convert PixelStack to solver-specific geometry representation.

        Args:
            pixel_stack: Solver-agnostic pixel stack structure.
        """

    @abstractmethod
    def setup_source(self, source_config: dict) -> None:
        """Configure excitation source.

        Args:
            source_config: Source configuration dictionary.
        """

    @abstractmethod
    def run(self) -> SimulationResult:
        """Execute simulation and return standardized results."""

    @abstractmethod
    def get_field_distribution(
        self,
        component: str,
        plane: str,
        position: float,
    ) -> np.ndarray:
        """Extract 2D field slice.

        Args:
            component: Field component ("Ex", "Ey", "Ez", "|E|2").
            plane: Slice plane ("xy", "xz", "yz").
            position: Position along the normal axis in um.

        Returns:
            2D field array.
        """

    def validate_energy_balance(
        self,
        result: SimulationResult,
        tolerance: float = 0.01,
    ) -> bool:
        """Check R + T + A ≈ 1.

        Args:
            result: Simulation result to validate.
            tolerance: Acceptable deviation from 1.0.

        Returns:
            True if energy is conserved within tolerance.
        """
        if result.reflection is None or result.transmission is None:
            return True

        if result.absorption is not None:
            total = result.reflection + result.transmission + result.absorption
        else:
            total = result.reflection + result.transmission

        max_violation = np.max(np.abs(total - 1.0))
        if max_violation > tolerance:
            logger.warning(
                f"Energy conservation violation: max |R+T+A-1| = {max_violation:.4f} "
                f"(tolerance: {tolerance})"
            )
            return False
        return True

    def run_timed(self) -> SimulationResult:
        """Run simulation with timing metadata."""
        t0 = time.perf_counter()
        result = self.run()
        elapsed = time.perf_counter() - t0
        result.metadata["runtime_seconds"] = elapsed
        result.metadata["solver_name"] = self.name
        result.metadata["solver_type"] = self.solver_type
        result.metadata["device"] = self.device
        logger.info(f"{self.name}: simulation completed in {elapsed:.2f}s")
        return result


class SolverFactory:
    """Factory for creating solver instances by name."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, solver_class: type) -> None:
        """Register a solver class."""
        cls._registry[name] = solver_class

    @classmethod
    def create(cls, name: str, config: dict, device: str = "cpu") -> SolverBase:
        """Create a solver instance by name.

        Args:
            name: Solver name (e.g. "torcwa", "grcwa", "fdtd_flaport").
            config: Solver config dict.
            device: Compute device.

        Returns:
            Solver instance.
        """
        if name not in cls._registry:
            # Try lazy import
            cls._try_import(name)

        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown solver: '{name}'. Available: {available}. "
                f"Make sure the solver package is installed."
            )

        solver: SolverBase = cls._registry[name](config, device)
        return solver

    @classmethod
    def _try_import(cls, name: str) -> None:
        """Try to import solver module to trigger registration."""
        import_map = {
            "torcwa": "compass.solvers.rcwa.torcwa_solver",
            "grcwa": "compass.solvers.rcwa.grcwa_solver",
            "meent": "compass.solvers.rcwa.meent_solver",
            "fmmax": "compass.solvers.rcwa.fmmax_solver",
            "fdtd_flaport": "compass.solvers.fdtd.flaport_solver",
            "fdtdz": "compass.solvers.fdtd.fdtdz_solver",
            "fdtdx": "compass.solvers.fdtd.fdtdx_solver",
            "meep": "compass.solvers.fdtd.meep_solver",
            "tmm": "compass.solvers.tmm.tmm_solver",
        }
        module_name = import_map.get(name)
        if module_name:
            try:
                __import__(module_name)
            except ImportError as e:
                logger.warning(f"Cannot import solver '{name}': {e}")

    @classmethod
    def list_solvers(cls) -> list:
        """List registered solver names."""
        return list(cls._registry.keys())
