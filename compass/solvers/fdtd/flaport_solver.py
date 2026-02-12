"""flaport/fdtd solver adapter for COMPASS.

Wraps the flaport fdtd library (Python-native 3D FDTD with PyTorch backend).
"""

from __future__ import annotations

import logging

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)


class FlaportFdtdSolver(SolverBase):
    """flaport/fdtd 3D FDTD solver adapter.

    Converts PixelStack to fdtd.Grid with objects, sources, and detectors.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._last_grid: object = None
        self._last_eps_3d: np.ndarray | None = None

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")
        self._pixel_stack = pixel_stack
        logger.info(f"fdtd_flaport: geometry setup for {pixel_stack.unit_cell} unit cell")

    def setup_source(self, source_config: dict) -> None:
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config
        logger.info(f"fdtd_flaport: source setup - {self._source.n_wavelengths} wavelengths")

    def run(self) -> SimulationResult:
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        try:
            import fdtd
        except ImportError as err:
            raise ImportError("fdtd is required. Install with: pip install fdtd") from err

        params = self.config.get("params", {})
        grid_spacing = params.get("grid_spacing", 0.02)  # um
        runtime = params.get("runtime", 200)  # femtoseconds
        pml_layers = params.get("pml_layers", 15)

        # Set backend
        if self.device == "cuda":
            fdtd.set_backend("torch.cuda")
        elif self.device == "cpu":
            fdtd.set_backend("torch")
        else:
            fdtd.set_backend("numpy")

        lx, ly = self._pixel_stack.domain_size
        z_min, z_max = self._pixel_stack.z_range
        total_z = z_max - z_min

        # Grid dimensions
        grid_spacing_m = grid_spacing * 1e-6  # convert to meters
        nx = int(lx / grid_spacing) + 1
        ny = int(ly / grid_spacing) + 1
        nz = int(total_z / grid_spacing) + 1 + 2 * pml_layers

        pol_runs = self._source.get_polarization_runs()
        all_qe: dict[str, list[float]] = {}
        all_R, all_T, all_A = [], [], []

        for wl_idx, wavelength in enumerate(self._source.wavelengths):
            logger.debug(
                f"fdtd_flaport: wavelength {wavelength:.4f} um ({wl_idx + 1}/{self._source.n_wavelengths})"
            )

            R_pol, T_pol, A_pol = [], [], []
            qe_pol_accum: dict[str, list[float]] = {}

            for pol in pol_runs:
                eps_3d = None
                try:
                    wavelength_m = wavelength * 1e-6
                    n_steps = int(runtime * 1e-15 / (grid_spacing_m / 3e8 / 2))

                    # --- Reference run (vacuum) ---
                    grid_ref = self._build_grid(
                        fdtd,
                        nx,
                        ny,
                        nz,
                        grid_spacing_m,
                        pml_layers,
                        wavelength_m,
                        pol,
                        eps_3d=None,
                    )
                    grid_ref.run(n_steps, progress_bar=False)
                    E_top_ref = self._get_detector_E(grid_ref, 0, n_steps)
                    E_bot_ref = self._get_detector_E(grid_ref, 1, n_steps)

                    # --- Structure run ---
                    eps_3d = self._pixel_stack.get_permittivity_grid(
                        wavelength, nx, ny, nz - 2 * pml_layers
                    )
                    grid_str = self._build_grid(
                        fdtd,
                        nx,
                        ny,
                        nz,
                        grid_spacing_m,
                        pml_layers,
                        wavelength_m,
                        pol,
                        eps_3d=eps_3d,
                    )

                    # Store for field extraction
                    self._last_grid = grid_str
                    self._last_eps_3d = eps_3d

                    grid_str.run(n_steps, progress_bar=False)
                    E_top_str = self._get_detector_E(grid_str, 0, n_steps)
                    E_bot_str = self._get_detector_E(grid_str, 1, n_steps)

                    # --- Compute R, T, A via reference normalization ---
                    P_inc = self._time_avg_intensity(E_bot_ref)
                    if P_inc > 0:
                        E_scattered = E_top_str - E_top_ref
                        R = self._time_avg_intensity(E_scattered) / P_inc
                        T = self._time_avg_intensity(E_bot_str) / P_inc
                    else:
                        R, T = 0.0, 0.0
                    A = float(np.clip(1.0 - R - T, 0.0, 1.0))

                except Exception as e:
                    logger.error(f"fdtd_flaport failed at λ={wavelength:.4f}um: {e}")
                    R, T, A = 0.0, 0.0, 0.0

                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)

                # Per-pixel QE via eps_imag weighting in PD regions
                if eps_3d is not None:
                    pixel_qe = self._compute_per_pixel_qe(eps_3d, wavelength, A)
                else:
                    pixel_qe = {}
                for key, val in pixel_qe.items():
                    qe_pol_accum.setdefault(key, []).append(val)

            n_pol = len(pol_runs)
            all_R.append(sum(R_pol) / n_pol)
            all_T.append(sum(T_pol) / n_pol)
            all_A.append(sum(A_pol) / n_pol)

            for k, vals in qe_pol_accum.items():
                all_qe.setdefault(k, []).append(sum(vals) / n_pol)

        result_arrays = {
            "reflection": np.array(all_R),
            "transmission": np.array(all_T),
            "absorption": np.array(all_A),
        }
        for arr_name, arr in result_arrays.items():
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                import warnings

                warnings.warn(f"fdtd_flaport: NaN/Inf detected in {arr_name} output", stacklevel=2)

        return SimulationResult(
            qe_per_pixel={k: np.array(v) for k, v in all_qe.items()},
            wavelengths=self._source.wavelengths,
            **result_arrays,
            metadata={
                "solver_name": "fdtd_flaport",
                "grid_spacing": grid_spacing,
                "device": self.device,
            },
        )

    def _build_grid(
        self,
        fdtd,
        nx: int,
        ny: int,
        nz: int,
        grid_spacing_m: float,
        pml_layers: int,
        wavelength_m: float,
        pol: str,
        eps_3d: np.ndarray | None = None,
    ):
        """Build an fdtd.Grid with boundaries, source, and detectors.

        Args:
            fdtd: The fdtd module.
            nx, ny, nz: Grid dimensions.
            grid_spacing_m: Grid spacing in meters.
            pml_layers: Number of PML layers.
            wavelength_m: Source wavelength in meters.
            pol: Polarization ("TE" or "TM").
            eps_3d: 3D permittivity array. None for vacuum reference run.

        Returns:
            Configured fdtd.Grid ready for simulation.
        """
        grid = fdtd.Grid(
            shape=(nx, ny, nz),
            grid_spacing=grid_spacing_m,
            permittivity=1.0,
        )

        # Boundaries: PML on z (slice indexing required by fdtd 0.3.5), periodic on x,y
        grid[0, :, :] = fdtd.PeriodicBoundary(name="x_boundary")
        grid[:, 0, :] = fdtd.PeriodicBoundary(name="y_boundary")
        grid[:, :, 0:pml_layers] = fdtd.PML(name="z_pml_low")
        grid[:, :, -pml_layers:] = fdtd.PML(name="z_pml_high")

        # Set permittivity if structure run
        if eps_3d is not None:
            eps_real = np.real(eps_3d).transpose(1, 0, 2)
            eps_real_safe = np.where(np.abs(eps_real) > 1e-30, eps_real, 1.0)
            grid.inverse_permittivity[:, :, pml_layers:-pml_layers] = 1.0 / eps_real_safe

        # Source — placed just inside PML boundary
        grid[:, :, pml_layers + 2] = fdtd.PlaneSource(
            period=wavelength_m / 3e8,
            polarization="x" if pol == "TE" else "y",
            name="source",
        )

        # Detectors: top (reflection side) and bottom (transmission side)
        grid[:, :, pml_layers + 5] = fdtd.BlockDetector(name="detector_top")
        grid[:, :, -pml_layers - 5] = fdtd.BlockDetector(name="detector_bot")

        return grid

    @staticmethod
    def _get_detector_E(grid, detector_idx: int, n_steps: int) -> np.ndarray:
        """Extract E-field time series from a BlockDetector.

        Returns:
            Array of shape (n_timesteps, nx, ny, 3) or similar detector output.
        """
        det = grid.detectors[detector_idx]
        E = det.detector_values()["E"]
        return np.array(E)

    @staticmethod
    def _time_avg_intensity(E: np.ndarray) -> float:
        """Compute time-averaged |E|^2 over the last 1/3 of timesteps.

        Args:
            E: Detector E-field array with time as the first axis.

        Returns:
            Spatially and temporally averaged intensity.
        """
        n_t = E.shape[0]
        start = max(0, n_t - n_t // 3)
        E_steady = E[start:]
        intensity = np.mean(np.sum(np.abs(E_steady) ** 2, axis=-1))
        return float(intensity)

    def _compute_per_pixel_qe(
        self,
        eps_3d: np.ndarray,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
        """Compute per-pixel QE from volume absorption in photodiode regions.

        Uses eps_imag to weight the absorption distribution among pixels.
        """
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        bayer = self._pixel_stack.bayer_map
        n_rows, n_cols = self._pixel_stack.unit_cell
        n_pixels = n_rows * n_cols
        if n_pixels == 0:
            return {}
        lx, ly = self._pixel_stack.domain_size
        z_min, z_max = self._pixel_stack.z_range
        nx, ny, nz = eps_3d.shape

        pixel_weights = {}
        total_weight = 0.0

        for pd in self._pixel_stack.photodiodes:
            r, c = pd.pixel_index
            color = pd.color
            key = f"{color}_{r}_{c}"

            # Map PD bounds to grid indices
            ix_min = max(0, int(((pd.position[0] - pd.size[0] / 2 + lx / 2) / lx) * nx))
            ix_max = min(nx, int(((pd.position[0] + pd.size[0] / 2 + lx / 2) / lx) * nx))
            iy_min = max(0, int(((pd.position[1] - pd.size[1] / 2 + ly / 2) / ly) * ny))
            iy_max = min(ny, int(((pd.position[1] + pd.size[1] / 2 + ly / 2) / ly) * ny))
            iz_min = max(0, int(((pd.position[2] - pd.size[2] / 2 - z_min) / (z_max - z_min)) * nz))
            iz_max = min(
                nz, int(((pd.position[2] + pd.size[2] / 2 - z_min) / (z_max - z_min)) * nz)
            )

            if ix_max <= ix_min or iy_max <= iy_min or iz_max <= iz_min:
                pixel_weights[key] = 0.0
                continue

            region = eps_3d[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max]
            weight = float(np.sum(np.imag(region)))
            weight = max(weight, 0.0)
            pixel_weights[key] = weight
            total_weight += weight

        qe_per_pixel = {}
        if total_weight > 0:
            for key, w in pixel_weights.items():
                qe_per_pixel[key] = total_absorption * (w / total_weight)
        else:
            for r in range(n_rows):
                for c in range(n_cols):
                    color = bayer[r][c]
                    key = f"{color}_{r}_{c}"
                    qe_per_pixel[key] = total_absorption / n_pixels

        return qe_per_pixel

    def get_field_distribution(
        self,
        component: str = "|E|2",
        plane: str = "xz",
        position: float = 0.0,
    ) -> np.ndarray:
        """Extract 2D field slice from last FDTD simulation.

        When the grid is available from the last run, extracts the E-field
        distribution. Otherwise builds an approximate field from the
        permittivity profile using Beer-Lambert absorption modeling.

        Args:
            component: Field component ("Ex", "Ey", "Ez", "|E|2").
            plane: Slice plane ("xy", "xz", "yz").
            position: Position along the normal axis in um.

        Returns:
            2D field array.
        """
        # Try extracting from last grid
        if self._last_grid is not None:
            try:
                grid = self._last_grid
                E = grid.E  # type: ignore[attr-defined]
                if E is not None:
                    E_np = np.array(E)  # (nx, ny, nz, 3)
                    return self._slice_field(E_np, component, plane, position)
            except Exception as e:
                logger.debug(f"fdtd_flaport: grid field extraction failed: {e}")

        # Fallback: approximate from permittivity
        if self._last_eps_3d is not None:
            return self._approximate_field_from_eps(self._last_eps_3d, component, plane, position)

        logger.warning("fdtd_flaport: no simulation data, returning zeros")
        return np.zeros((64, 64))

    def _slice_field(
        self,
        E: np.ndarray,
        component: str,
        plane: str,
        position: float,
    ) -> np.ndarray:
        """Slice a 4D field array (nx, ny, nz, 3) along the given plane."""
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        nx, ny, nz = E.shape[:3]
        lx, ly = self._pixel_stack.domain_size
        z_min, z_max = self._pixel_stack.z_range

        if component == "|E|2":
            field = np.sum(np.abs(E) ** 2, axis=-1)
        elif component == "Ex":
            field = np.abs(E[..., 0]) ** 2
        elif component == "Ey":
            field = np.abs(E[..., 1]) ** 2
        elif component == "Ez":
            field = np.abs(E[..., 2]) ** 2
        else:
            field = np.sum(np.abs(E) ** 2, axis=-1)

        if plane == "xy":
            z_idx = int(((position - z_min) / (z_max - z_min)) * nz)
            z_idx = max(0, min(nz - 1, z_idx))
            return np.asarray(field[:, :, z_idx])
        elif plane == "xz":
            y_idx = int(((position + ly / 2) / ly) * ny)
            y_idx = max(0, min(ny - 1, y_idx))
            return np.asarray(field[:, y_idx, :])
        elif plane == "yz":
            x_idx = int(((position + lx / 2) / lx) * nx)
            x_idx = max(0, min(nx - 1, x_idx))
            return np.asarray(field[x_idx, :, :])
        return np.zeros((64, 64))

    def _approximate_field_from_eps(
        self,
        eps_3d: np.ndarray,
        component: str,
        plane: str,
        position: float,
    ) -> np.ndarray:
        """Approximate field from permittivity using Beer-Lambert decay."""
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        nx, ny, nz = eps_3d.shape
        lx, ly = self._pixel_stack.domain_size

        if component == "|E|2" or component in ("Ex", "Ey", "Ez"):
            # Model intensity as exponential decay through absorbing layers
            k0 = 2 * np.pi / 0.55  # Default wavelength
            alpha = 2 * k0 * np.abs(np.imag(eps_3d))
            intensity = np.ones_like(alpha, dtype=float)
            dz = (self._pixel_stack.z_range[1] - self._pixel_stack.z_range[0]) / nz
            for zi in range(1, nz):
                intensity[:, :, zi] = intensity[:, :, zi - 1] * np.exp(-alpha[:, :, zi] * dz)
        else:
            intensity = np.real(eps_3d)

        if plane == "xy":
            z_idx = int(
                position / ((self._pixel_stack.z_range[1] - self._pixel_stack.z_range[0]) / nz)
            )
            z_idx = max(0, min(nz - 1, z_idx))
            return np.asarray(intensity[:, :, z_idx])
        elif plane == "xz":
            y_idx = int(((position + ly / 2) / ly) * ny)
            y_idx = max(0, min(ny - 1, y_idx))
            return np.asarray(intensity[:, y_idx, :])
        elif plane == "yz":
            x_idx = int(((position + lx / 2) / lx) * nx)
            x_idx = max(0, min(nx - 1, x_idx))
            return np.asarray(intensity[x_idx, :, :])
        return np.zeros((64, 64))


SolverFactory.register("fdtd_flaport", FlaportFdtdSolver)
