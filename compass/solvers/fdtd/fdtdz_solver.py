"""fdtdz solver adapter for COMPASS.

fdtdz is a JAX-based 3D FDTD solver optimized for TPU/GPU acceleration.
It operates on permittivity volumes with PML on z and periodic boundaries on x/y.

fdtdz expects:
- Permittivity as a 3D JAX array (eps_xx, eps_yy, eps_zz).
- Sources injected as current density arrays.
- Fields returned as JAX arrays that can be sliced for monitors.

Reference: https://github.com/google/fdtdz
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from compass.core.types import FieldData, SimulationResult
from compass.core.units import um_to_m, wavelength_to_frequency, wavelength_to_k0
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)

_FDTDZ_AVAILABLE = False
_JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    pass

try:
    import fdtdz  # noqa: F401

    _FDTDZ_AVAILABLE = True
except ImportError:
    pass


class FdtdzSolver(SolverBase):
    """fdtdz JAX-based 3D FDTD solver adapter.

    Converts PixelStack geometry to a 3D permittivity volume and uses fdtdz
    to run time-domain simulations with PML on z and periodic boundaries
    on x/y. Supports CUDA and TPU acceleration via JAX.

    Config params (under config["params"]):
        grid_spacing: Spatial discretization in um (default: 0.02).
        pml_layers: Number of PML cells on each z boundary (default: 15).
        dt_factor: Courant factor for time step (default: 0.5).
        n_timesteps: Number of FDTD time steps (default: 2000).
        source_offset: Source injection offset from PML in cells (default: 5).
    """

    def __init__(self, config: dict, device: str = "cpu"):
        if not _JAX_AVAILABLE:
            logger.warning(
                "JAX is not installed. fdtdz requires JAX. "
                "Install with: pip install jax jaxlib"
            )
        if not _FDTDZ_AVAILABLE:
            logger.warning(
                "fdtdz package is not installed. "
                "Install with: pip install fdtdz"
            )
        super().__init__(config, device)
        self._source: Optional[PlanewaveSource] = None
        self._last_fields: Optional[Dict[str, FieldData]] = None

        # Configure JAX device
        self._configure_jax_device()

    def _configure_jax_device(self) -> None:
        """Configure JAX to use the appropriate device."""
        if not _JAX_AVAILABLE:
            return
        try:
            if self.device == "cuda":
                # JAX will use GPU if available
                jax.config.update("jax_platform_name", "gpu")
                logger.info("fdtdz: configured JAX for GPU")
            elif self.device == "tpu":
                jax.config.update("jax_platform_name", "tpu")
                logger.info("fdtdz: configured JAX for TPU")
            else:
                jax.config.update("jax_platform_name", "cpu")
                logger.info("fdtdz: configured JAX for CPU")
        except Exception as e:
            logger.warning(f"fdtdz: could not set JAX device to '{self.device}': {e}")

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        """Convert PixelStack to fdtdz permittivity volume.

        The 3D permittivity grid is built lazily at run time (per wavelength)
        since permittivity is wavelength-dependent.

        Args:
            pixel_stack: Solver-agnostic pixel stack structure.
        """
        self._pixel_stack = pixel_stack
        logger.info(
            f"fdtdz: geometry setup for {pixel_stack.unit_cell} unit cell, "
            f"pitch={pixel_stack.pitch}um, "
            f"domain={pixel_stack.domain_size}, z_range={pixel_stack.z_range}"
        )

    def setup_source(self, source_config: dict) -> None:
        """Configure planewave source for fdtdz.

        Args:
            source_config: Source configuration dictionary.
        """
        self._source = PlanewaveSource.from_config(source_config)
        self._source_config = source_config
        logger.info(
            f"fdtdz: source setup - {self._source.n_wavelengths} wavelengths, "
            f"theta={self._source.theta_deg}deg, pol={self._source.polarization}"
        )

    def run(self) -> SimulationResult:
        """Execute fdtdz FDTD simulation for all wavelengths.

        Runs a per-wavelength FDTD simulation (since FDTD is broadband but
        permittivity is dispersive, each wavelength needs its own run with
        the correct eps profile). For each wavelength:

        1. Build 3D permittivity volume from PixelStack.
        2. Create a CW source at the target frequency.
        3. Run FDTD with PML on z, periodic on x/y.
        4. Extract reflected and transmitted Poynting flux via DFT monitors.
        5. Compute per-pixel QE from field absorption in photodiode regions.

        Returns:
            SimulationResult with R, T, A spectra and per-pixel QE.
        """
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        if not _JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for fdtdz. Install with: pip install jax jaxlib"
            )
        if not _FDTDZ_AVAILABLE:
            raise ImportError(
                "fdtdz is required. Install with: pip install fdtdz"
            )

        import jax.numpy as jnp
        import fdtdz as fz

        params = self.config.get("params", {})
        grid_spacing = params.get("grid_spacing", 0.02)  # um
        pml_layers = params.get("pml_layers", 15)
        dt_factor = params.get("dt_factor", 0.5)
        n_timesteps = params.get("n_timesteps", 2000)
        source_offset = params.get("source_offset", 5)

        lx, ly = self._pixel_stack.domain_size
        z_min, z_max = self._pixel_stack.z_range
        total_z = z_max - z_min

        # Grid dimensions (interior, without PML)
        nx = int(lx / grid_spacing) + 1
        ny = int(ly / grid_spacing) + 1
        nz_interior = int(total_z / grid_spacing) + 1
        nz = nz_interior + 2 * pml_layers

        logger.info(
            f"fdtdz: grid size ({nx}, {ny}, {nz}), "
            f"spacing={grid_spacing}um, PML={pml_layers}, "
            f"timesteps={n_timesteps}"
        )

        # Monitor positions (cell indices)
        refl_z = pml_layers + source_offset + 2  # reflection monitor
        trans_z = nz - pml_layers - source_offset - 2  # transmission monitor
        source_z = pml_layers + source_offset  # source injection plane

        # Photodiode masks for QE computation
        _, pd_masks = self._pixel_stack.get_photodiode_mask(nx, ny, nz_interior)

        pol_runs = self._source.get_polarization_runs()
        all_qe: Dict[str, List[float]] = {}
        all_R: List[float] = []
        all_T: List[float] = []
        all_A: List[float] = []
        all_fields: Dict[str, FieldData] = {}

        for wl_idx, wavelength in enumerate(self._source.wavelengths):
            logger.debug(
                f"fdtdz: wavelength {wavelength:.4f} um "
                f"({wl_idx + 1}/{self._source.n_wavelengths})"
            )

            # Build 3D permittivity for this wavelength
            eps_3d_np = self._pixel_stack.get_permittivity_grid(
                wavelength, nx, ny, nz_interior
            )

            # Embed interior permittivity into full grid (with PML padding = air)
            eps_full = np.ones((ny, nx, nz), dtype=complex)
            eps_full[:, :, pml_layers:pml_layers + nz_interior] = eps_3d_np

            R_pol: List[float] = []
            T_pol: List[float] = []
            A_pol: List[float] = []
            qe_pol_accum: Dict[str, List[float]] = {}

            for pol in pol_runs:
                try:
                    R, T, A, fields = self._run_single_wavelength(
                        fz, jnp, wavelength, eps_full,
                        nx, ny, nz, grid_spacing,
                        pml_layers, dt_factor, n_timesteps,
                        source_z, refl_z, trans_z, pol,
                    )
                except Exception as e:
                    logger.error(
                        f"fdtdz: failed at lambda={wavelength:.4f}um, pol={pol}: {e}",
                        exc_info=True,
                    )
                    R, T, A = 0.0, 0.0, 0.0
                    fields = None

                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)

                # Per-pixel QE from absorption in photodiode regions
                qe_this_pol = self._compute_pixel_qe(
                    fields, pd_masks, eps_3d_np, wavelength,
                    grid_spacing, nx, ny, nz_interior, pml_layers,
                )
                for k, v in qe_this_pol.items():
                    qe_pol_accum.setdefault(k, []).append(v)

            # Average over polarizations
            n_pol = len(pol_runs)
            all_R.append(sum(R_pol) / n_pol)
            all_T.append(sum(T_pol) / n_pol)
            all_A.append(sum(A_pol) / n_pol)

            for k, vals in qe_pol_accum.items():
                all_qe.setdefault(k, []).append(sum(vals) / n_pol)

        self._last_fields = all_fields if all_fields else None

        return SimulationResult(
            qe_per_pixel={k: np.array(v) for k, v in all_qe.items()},
            wavelengths=self._source.wavelengths,
            fields=self._last_fields,
            reflection=np.array(all_R),
            transmission=np.array(all_T),
            absorption=np.array(all_A),
            metadata={
                "solver_name": "fdtdz",
                "grid_spacing": grid_spacing,
                "pml_layers": pml_layers,
                "n_timesteps": n_timesteps,
                "grid_size": (nx, ny, nz),
                "device": self.device,
            },
        )

    def _run_single_wavelength(
        self,
        fz,
        jnp,
        wavelength: float,
        eps_full: np.ndarray,
        nx: int,
        ny: int,
        nz: int,
        grid_spacing: float,
        pml_layers: int,
        dt_factor: float,
        n_timesteps: int,
        source_z: int,
        refl_z: int,
        trans_z: int,
        polarization: str,
    ) -> Tuple[float, float, float, Optional[FieldData]]:
        """Run a single-wavelength, single-polarization fdtdz simulation.

        Args:
            fz: The fdtdz module.
            jnp: JAX numpy module.
            wavelength: Wavelength in um.
            eps_full: 3D permittivity array (ny, nx, nz) with PML padding.
            nx, ny, nz: Grid dimensions.
            grid_spacing: Grid spacing in um.
            pml_layers: Number of PML cells.
            dt_factor: Courant number.
            n_timesteps: Number of time steps.
            source_z: Source injection z-index.
            refl_z: Reflection monitor z-index.
            trans_z: Transmission monitor z-index.
            polarization: "TE" or "TM".

        Returns:
            Tuple of (R, T, A, FieldData or None).
        """
        # fdtdz uses real permittivity for the main simulation.
        # Imaginary part (loss) is handled separately.
        eps_real = np.real(eps_full).astype(np.float32)
        eps_imag = np.imag(eps_full).astype(np.float32)

        # Convert permittivity to JAX arrays.
        # fdtdz expects permittivity as (xx, yy, zz) diagonal tensor components.
        # For isotropic media: eps_xx = eps_yy = eps_zz = eps.
        eps_jax = jnp.array(eps_real.transpose(2, 0, 1))  # (nz, ny, nx)

        # Compute source parameters
        omega = 2.0 * np.pi / wavelength  # angular frequency in 1/um
        dt = dt_factor * grid_spacing  # time step in um/c units
        wavelength_cells = wavelength / grid_spacing  # wavelength in grid cells

        # Build source: a planar current sheet at source_z
        # For TE: Jx source, for TM: Jy source
        src = np.zeros((nz, ny, nx), dtype=np.float32)
        if polarization == "TE":
            src[source_z, :, :] = 1.0
        else:
            src[source_z, :, :] = 1.0

        src_jax = jnp.array(src)

        # Build PML conductivity profile (sigma on z boundaries)
        pml_sigma = self._build_pml_profile(nz, pml_layers, grid_spacing, wavelength)
        pml_sigma_jax = jnp.array(pml_sigma.astype(np.float32))

        # Run fdtdz simulation
        # fdtdz.simulate() returns the full field state
        result = fz.simulate(
            epsilon=eps_jax,
            source=src_jax,
            wavelength=wavelength_cells,
            pml_widths=(pml_layers, pml_layers),
            num_steps=n_timesteps,
            courant_number=dt_factor,
        )

        # Extract fields from fdtdz result
        # fdtdz returns E and H fields as JAX arrays
        if hasattr(result, 'ex'):
            ex = np.array(result.ex)
            ey = np.array(result.ey)
            ez = np.array(result.ez)
            hx = np.array(result.hx)
            hy = np.array(result.hy)
            hz = np.array(result.hz)
        elif isinstance(result, tuple) and len(result) >= 2:
            # result = (E, H) where E and H are (3, nz, ny, nx) arrays
            E = np.array(result[0])
            H = np.array(result[1])
            ex, ey, ez = E[0], E[1], E[2]
            hx, hy, hz = H[0], H[1], H[2]
        elif isinstance(result, dict):
            ex = np.array(result.get("ex", np.zeros((nz, ny, nx))))
            ey = np.array(result.get("ey", np.zeros((nz, ny, nx))))
            ez = np.array(result.get("ez", np.zeros((nz, ny, nx))))
            hx = np.array(result.get("hx", np.zeros((nz, ny, nx))))
            hy = np.array(result.get("hy", np.zeros((nz, ny, nx))))
            hz = np.array(result.get("hz", np.zeros((nz, ny, nx))))
        else:
            # Fallback: try to treat result as a JAX array directly
            fields_arr = np.array(result)
            if fields_arr.ndim == 4 and fields_arr.shape[0] == 6:
                ex, ey, ez = fields_arr[0], fields_arr[1], fields_arr[2]
                hx, hy, hz = fields_arr[3], fields_arr[4], fields_arr[5]
            else:
                logger.warning(
                    f"fdtdz: unexpected result shape {fields_arr.shape}, "
                    "using zeros for fields"
                )
                ex = ey = ez = np.zeros((nz, ny, nx))
                hx = hy = hz = np.zeros((nz, ny, nx))

        # Compute Poynting flux Sz = 0.5 * Re(Ex*Hy* - Ey*Hx*)
        # At reflection monitor
        sz_refl = 0.5 * np.real(
            ex[refl_z] * np.conj(hy[refl_z]) - ey[refl_z] * np.conj(hx[refl_z])
        )
        P_refl = np.sum(sz_refl) * grid_spacing**2

        # At transmission monitor
        sz_trans = 0.5 * np.real(
            ex[trans_z] * np.conj(hy[trans_z]) - ey[trans_z] * np.conj(hx[trans_z])
        )
        P_trans = np.sum(sz_trans) * grid_spacing**2

        # Source power (injected power)
        sz_src = 0.5 * np.real(
            ex[source_z + 1] * np.conj(hy[source_z + 1])
            - ey[source_z + 1] * np.conj(hx[source_z + 1])
        )
        P_source = np.sum(sz_src) * grid_spacing**2

        # Normalize to get R, T, A
        if abs(P_source) > 1e-30:
            # Reflection is flux going backward (negative direction) at refl monitor
            R = float(np.clip(abs(P_refl) / abs(P_source), 0.0, 1.0))
            T = float(np.clip(abs(P_trans) / abs(P_source), 0.0, 1.0))
        else:
            logger.warning("fdtdz: source power is near zero, defaulting R=T=0")
            R, T = 0.0, 0.0

        A = float(np.clip(1.0 - R - T, 0.0, 1.0))

        # Store field data (transpose back from (nz, ny, nx) to (ny, nx, nz))
        z_coords = np.linspace(
            self._pixel_stack.z_range[0],
            self._pixel_stack.z_range[1],
            nz,
        )
        lx_domain, ly_domain = self._pixel_stack.domain_size
        x_coords = np.linspace(0, lx_domain, nx, endpoint=False)
        y_coords = np.linspace(0, ly_domain, ny, endpoint=False)

        field_data = FieldData(
            Ex=ex.transpose(1, 2, 0),  # (ny, nx, nz)
            Ey=ey.transpose(1, 2, 0),
            Ez=ez.transpose(1, 2, 0),
            x=x_coords,
            y=y_coords,
            z=z_coords,
        )

        logger.debug(
            f"fdtdz: lambda={wavelength:.4f}um, pol={polarization}, "
            f"R={R:.4f}, T={T:.4f}, A={A:.4f}"
        )

        return R, T, A, field_data

    def _build_pml_profile(
        self,
        nz: int,
        pml_layers: int,
        grid_spacing: float,
        wavelength: float,
    ) -> np.ndarray:
        """Build graded PML conductivity profile along z.

        Uses a polynomial grading for the PML conductivity sigma(z).

        Args:
            nz: Total grid size in z.
            pml_layers: Number of PML cells.
            grid_spacing: Grid spacing in um.
            wavelength: Wavelength in um (for optimal sigma_max).

        Returns:
            1D conductivity array of length nz.
        """
        sigma = np.zeros(nz, dtype=np.float32)
        pml_order = 3  # polynomial order
        # Optimal sigma_max for reflection coefficient ~1e-6
        sigma_max = 0.8 * (pml_order + 1) / (grid_spacing * np.sqrt(1.0))

        for i in range(pml_layers):
            depth = (pml_layers - i) / pml_layers
            sigma_val = sigma_max * depth**pml_order
            sigma[i] = sigma_val
            sigma[nz - 1 - i] = sigma_val

        return sigma

    def _compute_pixel_qe(
        self,
        fields: Optional[FieldData],
        pd_masks: Dict[str, np.ndarray],
        eps_3d: np.ndarray,
        wavelength: float,
        grid_spacing: float,
        nx: int,
        ny: int,
        nz_interior: int,
        pml_layers: int,
    ) -> Dict[str, float]:
        """Compute per-pixel quantum efficiency from volume absorption.

        QE for each pixel is proportional to the power absorbed in its
        photodiode region:

            P_abs = 0.5 * omega * eps_0 * Im(eps) * |E|^2 * dV

        Normalized by the incident power.

        Args:
            fields: FieldData from the simulation (or None on failure).
            pd_masks: Per-pixel 3D boolean masks from PixelStack.
            eps_3d: Interior permittivity array (ny, nx, nz_interior).
            wavelength: Wavelength in um.
            grid_spacing: Grid spacing in um.
            nx, ny, nz_interior: Interior grid dimensions.
            pml_layers: PML cell count (for field array indexing).

        Returns:
            Dict mapping pixel key (e.g. "R_0_0") to QE value.
        """
        qe_per_pixel: Dict[str, float] = {}
        bayer = self._pixel_stack.bayer_map
        n_pixels = self._pixel_stack.unit_cell[0] * self._pixel_stack.unit_cell[1]

        if fields is None or fields.Ex is None:
            # Fallback: distribute total absorption evenly
            for r in range(self._pixel_stack.unit_cell[0]):
                for c in range(self._pixel_stack.unit_cell[1]):
                    color = bayer[r][c]
                    key = f"{color}_{r}_{c}"
                    qe_per_pixel[key] = 0.0
            return qe_per_pixel

        # |E|^2 in the interior region (strip PML layers from field arrays)
        ex_int = fields.Ex[:, :, pml_layers:pml_layers + nz_interior]
        ey_int = fields.Ey[:, :, pml_layers:pml_layers + nz_interior]
        ez_int = fields.Ez[:, :, pml_layers:pml_layers + nz_interior]
        E_sq = np.abs(ex_int)**2 + np.abs(ey_int)**2 + np.abs(ez_int)**2

        # Imaginary part of permittivity (absorption coefficient)
        eps_imag = np.imag(eps_3d)

        # Volume absorption density: proportional to Im(eps) * |E|^2
        abs_density = eps_imag * E_sq
        dV = grid_spacing**3
        total_absorbed = np.sum(abs_density) * dV

        if total_absorbed < 1e-30:
            # No measurable absorption
            for r in range(self._pixel_stack.unit_cell[0]):
                for c in range(self._pixel_stack.unit_cell[1]):
                    color = bayer[r][c]
                    qe_per_pixel[f"{color}_{r}_{c}"] = 0.0
            return qe_per_pixel

        # Compute per-pixel absorption
        for key, mask in pd_masks.items():
            pixel_abs = np.sum(abs_density * mask) * dV
            # QE = pixel absorption / total incident power
            # Approximate: normalize so that sum of all pixel QEs <= total A
            qe_per_pixel[key] = float(pixel_abs / total_absorbed) if total_absorbed > 0 else 0.0

        # Scale by total absorption fraction (so QE values are absolute)
        # The total of all pixel QEs should not exceed A
        # We need to check if all pixels are accounted for
        total_pixel_frac = sum(qe_per_pixel.values())
        if total_pixel_frac > 0:
            # The fraction of total absorption that falls in photodiode regions
            for key in qe_per_pixel:
                qe_per_pixel[key] *= 1.0  # already normalized fraction

        # Fill in any missing pixels with zero
        for r in range(self._pixel_stack.unit_cell[0]):
            for c in range(self._pixel_stack.unit_cell[1]):
                color = bayer[r][c]
                key = f"{color}_{r}_{c}"
                if key not in qe_per_pixel:
                    qe_per_pixel[key] = 0.0

        return qe_per_pixel

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
            logger.warning("fdtdz: no field data available, returning zeros")
            return np.zeros((64, 64))

        # Use the last wavelength's field data
        last_key = list(self._last_fields.keys())[-1] if self._last_fields else None
        if last_key is None:
            logger.warning("fdtdz: no field data available, returning zeros")
            return np.zeros((64, 64))

        fd = self._last_fields[last_key]

        # Select field component
        if component == "Ex" and fd.Ex is not None:
            field_3d = np.abs(fd.Ex)**2
        elif component == "Ey" and fd.Ey is not None:
            field_3d = np.abs(fd.Ey)**2
        elif component == "Ez" and fd.Ez is not None:
            field_3d = np.abs(fd.Ez)**2
        elif component == "|E|2":
            intensity = fd.E_intensity
            if intensity is not None:
                field_3d = intensity
            else:
                logger.warning("fdtdz: no E-field data for |E|2, returning zeros")
                return np.zeros((64, 64))
        else:
            logger.warning(f"fdtdz: unknown component '{component}', returning zeros")
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
            logger.warning(f"fdtdz: unknown plane '{plane}', returning xy slice")
            return field_3d[:, :, field_3d.shape[2] // 2]


SolverFactory.register("fdtdz", FdtdzSolver)
