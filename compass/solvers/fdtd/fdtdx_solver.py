"""FDTDX solver adapter for COMPASS.

FDTDX is a JAX-based, fully differentiable 3D FDTD solver with multi-GPU
support. It uses a declarative object-placement API to set up simulations
and runs time-stepping of Maxwell's equations on a Yee grid.

This adapter translates the COMPASS PixelStack into FDTDX's simulation
volume with appropriate material placement, planewave sources, and DFT
flux monitors.

Reference: https://github.com/ymahlau/fdtdx
License: MIT
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from compass.core.types import FieldData, SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)

_FDTDX_AVAILABLE = False
_JAX_AVAILABLE = False
try:
    import jax

    _JAX_AVAILABLE = True
except ImportError:
    pass

try:
    import fdtdx  # noqa: F401

    _FDTDX_AVAILABLE = True
except ImportError:
    pass


class FdtdxSolver(SolverBase):
    """FDTDX JAX-based 3D FDTD solver adapter.

    Converts PixelStack geometry to a 3D permittivity volume and uses FDTDX
    to run fully differentiable time-domain simulations. Supports multi-GPU
    execution via JAX's device parallelism.

    FDTDX computes in the time domain and uses DFT to extract
    frequency-domain reflection/transmission. Grid spacing should be
    approximately lambda/20 for accuracy. PML boundaries absorb outgoing
    waves on the z-axis while x/y boundaries are periodic.

    Config params (under config["params"]):
        grid_spacing: Spatial discretization in um (default: 0.02).
        pml_layers: Number of PML cells on each z boundary (default: 10).
        duration: Simulated time in um/c (default: auto from the stack's
            optical path, 6 * sum(n d) + 40 * lambda_max).
        time_steps: Alternative to duration (steps at the Courant time step).
        dtype: Float precision ("float32" or "float64", default: "float32").
        courant_factor: Courant number for time step stability (default: 0.5).
    """

    def _optical_path(self, wavelength: float) -> tuple[float, float]:
        """(sum Re(n) * d over the stack in um, max Re(n)) at ``wavelength``."""
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        path, n_max = 0.0, 1.0
        for s in self._pixel_stack.get_layer_slices(wavelength, 16, 16):
            n = float(np.max(np.real(np.sqrt(s.eps_grid))))
            path += n * s.thickness
            n_max = max(n_max, n)
        return path, n_max

    def _resolve_duration(self, params: dict, dx: float) -> float:
        """Simulated time in um/c.

        Explicit ``duration`` (um/c) or ``time_steps`` win. Otherwise the
        duration covers the source pulse plus several round trips of the
        stack's optical path: the old default (time_steps=2000, ~11.5 um/c
        at dx=0.02) truncated the response of thick silicon and changed R
        by up to 0.2.
        """
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")
        wl_min = float(np.min(self._source.wavelengths))
        wl_max = float(np.max(self._source.wavelengths))
        path, n_max = self._optical_path(wl_min)
        cells_per_wl = wl_min / (n_max * dx)
        if cells_per_wl < 10:
            logger.warning(
                f"fdtdx: grid_spacing={dx} um resolves λ/n = {wl_min / n_max:.3f} um with "
                f"only {cells_per_wl:.1f} cells (n_max={n_max:.2f}); expect R/T errors of "
                "several percent for high-index layers (aim for >= 15-20 cells)"
            )
        if params.get("duration") is not None:
            return float(params["duration"])
        if params.get("time_steps") is not None:
            return float(float(params["time_steps"]) * dx * self._courant_factor / np.sqrt(3))
        return float(6.0 * path + 40.0 * wl_max)

    def __init__(self, config: dict, device: str = "cpu"):
        if not _JAX_AVAILABLE:
            logger.warning(
                "JAX is not installed. FDTDX requires JAX. Install with: pip install jax jaxlib"
            )
        if not _FDTDX_AVAILABLE:
            logger.warning("fdtdx package is not installed. Install with: pip install fdtdx")
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._last_fields: dict[str, FieldData] | None = None

        params = config.get("params", {})
        self._grid_spacing: float = params.get("grid_spacing", 0.02)
        self._pml_layers: int = params.get("pml_layers", 10)
        self._time_steps: int = params.get("time_steps", 2000)
        self._dtype_str: str = params.get("dtype", "float32")
        self._courant_factor: float = params.get("courant_factor", 0.5)

        # Configure JAX device
        self._configure_jax_device()

    def _configure_jax_device(self) -> None:
        """Configure JAX to use the appropriate device."""
        if not _JAX_AVAILABLE:
            return
        try:
            if self.device.startswith("cuda"):
                jax.config.update("jax_platform_name", "gpu")
                logger.info("fdtdx: configured JAX for GPU")
            elif self.device == "tpu":
                jax.config.update("jax_platform_name", "tpu")
                logger.info("fdtdx: configured JAX for TPU")
            else:
                jax.config.update("jax_platform_name", "cpu")
                logger.info("fdtdx: configured JAX for CPU")
        except Exception as e:
            logger.warning(f"fdtdx: could not set JAX device to '{self.device}': {e}")

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        """Convert PixelStack to FDTDX geometry representation.

        The 3D permittivity grid is built lazily at run time (per wavelength)
        since permittivity is wavelength-dependent.

        Args:
            pixel_stack: Solver-agnostic pixel stack structure.
        """
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")
        self._pixel_stack = pixel_stack
        logger.info(
            f"fdtdx: geometry setup for {pixel_stack.unit_cell} unit cell, "
            f"pitch={pixel_stack.pitch}um, "
            f"domain={pixel_stack.domain_size}, z_range={pixel_stack.z_range}"
        )

    def setup_source(self, source_config: dict) -> None:
        """Configure planewave source for FDTDX.

        Args:
            source_config: Source configuration dictionary.
        """
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.theta_deg != 0.0:
            raise NotImplementedError("fdtdx supports normal incidence only")
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config
        logger.info(
            f"fdtdx: source setup - {self._source.n_wavelengths} wavelengths, "
            f"theta={self._source.theta_deg}deg, pol={self._source.polarization}"
        )

    def run(self) -> SimulationResult:
        """Execute FDTDX FDTD simulation for all wavelengths.

        Runs a per-wavelength FDTD simulation (since FDTD is broadband but
        permittivity is dispersive, each wavelength needs its own run with
        the correct eps profile). For each wavelength:

        1. Build 3D permittivity volume from PixelStack.
        2. Set up FDTDX simulation volume with PML boundaries.
        3. Place materials (permittivity grid) in the volume.
        4. Add planewave source and DFT monitors.
        5. Run FDTD time-stepping.
        6. Extract R, T from flux monitors and compute A = 1 - R - T.
        Regional QE is unavailable until volume absorption is independently calibrated.

        Returns:
            SimulationResult with R, T, A spectra and per-pixel QE.
        """
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        if not _JAX_AVAILABLE:
            raise ImportError("JAX is required for fdtdx. Install with: pip install jax jaxlib")
        if not _FDTDX_AVAILABLE:
            raise ImportError("fdtdx is required. Install with: pip install fdtdx")

        from compass.solvers.fdtd.fdtdx_native import solve

        params = self.config.get("params", {})
        lx, ly = self._pixel_stack.domain_size
        lo, hi = self._pixel_stack.z_range
        # Snap the cell size so the periodic cell is exactly lx x ly.
        nx = max(1, round(lx / self._grid_spacing))
        dx = lx / nx
        ny = max(1, round(ly / dx))
        if abs(ny * dx - ly) > 1e-6 * ly:
            raise ValueError(
                f"domain {lx} x {ly} um has no common cell size near {self._grid_spacing} um"
            )
        nz = max(1, round((hi - lo) / dx))
        duration = self._resolve_duration(params, dx)
        self._failed_runs = []
        spectra = []
        observed = None
        for wl in self._source.wavelengths:
            eps = self._pixel_stack.get_permittivity_grid(wl, nx, ny, nz)
            values = []
            for pol in self._source.get_polarization_runs():
                try:
                    R, T, A, observed = solve(
                        eps,
                        float(wl),
                        dx,
                        duration,
                        self._pml_layers,
                        self.device,
                        pol,
                        courant_factor=self._courant_factor,
                        precision="float32" if self._dtype_str == "float32" else "float64",
                    )
                    if not np.isfinite([R, T, A]).all():
                        raise RuntimeError("FDTDX returned NaN/Inf")
                    values.append([R, T, A])
                except Exception as exc:
                    self._record_failed_run(wl, pol, exc)
                    values.append([np.nan] * 3)
            spectra.append(np.mean(values, axis=0))
        rta = np.asarray(spectra)
        if not np.isfinite(rta).all():
            warnings.warn("fdtdx: NaN/Inf detected in R/T/A output", stacklevel=2)
        # Regional QE requires a separately calibrated volume absorption monitor.
        qe = {key: np.full(len(rta), np.nan) for key in self._nan_pixel_qe()}
        return SimulationResult(
            qe_per_pixel=qe,
            wavelengths=self._source.wavelengths,
            reflection=rta[:, 0],
            transmission=rta[:, 1],
            absorption=rta[:, 2],
            metadata={
                "solver_name": "fdtdx",
                "actual_device": observed,
                "qe_method": "unavailable",
                "grid_spacing": dx,
                "pml_layers": self._pml_layers,
                "interior_grid_size": [nx, ny, nz],
                "grid_size": [nx, ny, nz + 2 * (max(2, round(0.8 / dx)) + self._pml_layers)],
                "air_padding_um": max(2, round(0.8 / dx)) * dx,
                "duration": duration,
                "time_steps": round(duration / (dx * self._courant_factor / np.sqrt(3))),
                "courant_factor": self._courant_factor,
                "device": self.device,
                "api": "place_objects/run_fdtd/PhasorPoyntingFluxDetector",
                **self._failure_metadata(),
            },
        )

    def get_field_distribution(
        self,
        component: str = "|E|2",
        plane: str = "xz",
        position: float = 0.0,
    ) -> np.ndarray:
        """Extract 2D field slice from the last simulation.

        Args:
            component: Field component ("Ex", "Ey", "Ez", "|E|2").
            plane: Slice plane ("xy", "xz", "yz").
            position: Position along the normal axis in um.

        Returns:
            2D field array.
        """
        if self._last_fields is None:
            logger.warning("fdtdx: no field data available, returning zeros")
            return np.zeros((64, 64))

        # Use the last wavelength's field data
        last_key = list(self._last_fields.keys())[-1] if self._last_fields else None
        if last_key is None:
            logger.warning("fdtdx: no field data available, returning zeros")
            return np.zeros((64, 64))

        fd = self._last_fields[last_key]

        # Select field component
        if component == "Ex" and fd.Ex is not None:
            field_3d = np.abs(fd.Ex) ** 2
        elif component == "Ey" and fd.Ey is not None:
            field_3d = np.abs(fd.Ey) ** 2
        elif component == "Ez" and fd.Ez is not None:
            field_3d = np.abs(fd.Ez) ** 2
        elif component == "|E|2":
            intensity = fd.E_intensity
            if intensity is not None:
                field_3d = intensity
            else:
                logger.warning("fdtdx: no E-field data for |E|2, returning zeros")
                return np.zeros((64, 64))
        else:
            logger.warning(f"fdtdx: unknown component '{component}', returning zeros")
            return np.zeros((64, 64))

        # Extract 2D slice
        return self._extract_slice(field_3d, fd, plane, position)

    def _extract_slice(
        self,
        field_3d: np.ndarray,
        fd: FieldData,
        plane: str,
        position: float,
    ) -> np.ndarray:
        """Extract a 2D slice from a 3D field array.

        Args:
            field_3d: 3D field array (ny, nx, nz).
            fd: FieldData with coordinate arrays.
            plane: Slice plane ("xy", "xz", "yz").
            position: Position along the normal axis in um.

        Returns:
            2D slice array.
        """
        if plane == "xy":
            # Slice at fixed z
            if fd.z is not None:
                idx = int(np.argmin(np.abs(fd.z - position)))
            else:
                idx = field_3d.shape[2] // 2
            return field_3d[:, :, idx]

        elif plane == "xz":
            # Slice at fixed y
            if fd.y is not None:
                idx = int(np.argmin(np.abs(fd.y - position)))
            else:
                idx = field_3d.shape[0] // 2
            return field_3d[idx, :, :]

        elif plane == "yz":
            # Slice at fixed x
            if fd.x is not None:
                idx = int(np.argmin(np.abs(fd.x - position)))
            else:
                idx = field_3d.shape[1] // 2
            return field_3d[:, idx, :]

        else:
            logger.warning(f"fdtdx: unknown plane '{plane}', returning xy slice")
            return field_3d[:, :, field_3d.shape[2] // 2]


SolverFactory.register("fdtdx", FdtdxSolver)
