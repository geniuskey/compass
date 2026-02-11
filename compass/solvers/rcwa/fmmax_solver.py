"""fmmax (Fourier Modal Method with JAX) RCWA solver adapter for COMPASS.

Wraps the fmmax library, which implements the Fourier Modal Method (FMM)
using JAX for automatic differentiation and GPU acceleration.
Supports multiple FMM formulations: Pol, Normal, Jones, JonesDirect.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)

# Valid FMM formulation names mapping to fmmax enums
_FORMULATION_MAP = {
    "pol": "POL",
    "normal": "NORMAL",
    "jones": "JONES",
    "jonesdirect": "JONES_DIRECT",
}


class FmmaxSolver(SolverBase):
    """fmmax RCWA solver adapter.

    Uses the fmmax library (JAX-based Fourier Modal Method) for
    rigorous coupled-wave analysis of layered periodic structures.

    Config params:
        fourier_order: [int, int] -- Number of Fourier harmonics in x, y.
        fmm_formulation: str -- FMM vector formulation type
            ("pol", "normal", "jones", "jonesdirect"). Default: "jones".
        dtype: str -- Complex dtype for computation ("complex64" or "complex128").
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._last_layer_slices: list | None = None
        self._last_wavelength: float | None = None

        params = config.get("params", {})
        self._fmm_formulation = params.get("fmm_formulation", "jones")
        self._dtype_str = params.get("dtype", "complex64")

        # Validate formulation name early
        if self._fmm_formulation.lower() not in _FORMULATION_MAP:
            raise ValueError(
                f"Unknown fmm_formulation: '{self._fmm_formulation}'. "
                f"Valid options: {list(_FORMULATION_MAP.keys())}"
            )

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        """Convert PixelStack to solver-specific geometry representation.

        Args:
            pixel_stack: Solver-agnostic pixel stack structure.

        Raises:
            ValueError: If pixel_stack is None or has no layers.
        """
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")
        self._pixel_stack = pixel_stack
        logger.info(
            f"fmmax: geometry setup, formulation={self._fmm_formulation}"
        )

    def setup_source(self, source_config: dict) -> None:
        """Configure excitation source.

        Args:
            source_config: Source configuration dictionary.

        Raises:
            ValueError: If wavelengths are empty or non-positive.
        """
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config
        logger.info(
            f"fmmax: source setup - {self._source.n_wavelengths} wavelengths"
        )

    def run(self) -> SimulationResult:
        """Execute fmmax RCWA simulation and return standardized results.

        Returns:
            SimulationResult with QE, R/T/A spectra, and metadata.

        Raises:
            RuntimeError: If setup_geometry() or setup_source() not called.
            ImportError: If fmmax or jax are not installed.
        """
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        try:
            import jax
            import jax.numpy as jnp
            from fmmax import basis, fmm
        except ImportError as err:
            raise ImportError(
                "fmmax and jax are required. "
                "Install with: pip install fmmax jax"
            ) from err

        # Configure JAX device
        if self.device == "cpu":
            jax.config.update("jax_platform_name", "cpu")

        params = self.config.get("params", {})
        fourier_order = params.get("fourier_order", [9, 9])

        # Resolve FMM formulation enum
        formulation_key = _FORMULATION_MAP[self._fmm_formulation.lower()]
        formulation = getattr(fmm.Formulation, formulation_key)

        # Resolve dtype
        dtype_map = {
            "complex64": jnp.complex64,
            "complex128": jnp.complex128,
        }
        jax_dtype = dtype_map.get(self._dtype_str, jnp.complex64)

        lx, ly = self._pixel_stack.domain_size
        # Grid resolution: at least 64, or 3x the Fourier order
        nx = max(64, (2 * fourier_order[0] + 1) * 3)
        ny = max(64, (2 * fourier_order[1] + 1) * 3)

        # Expansion for Fourier basis
        expansion = basis.generate_expansion(
            primitive_lattice_vectors=basis.LatticeVectors(
                u=jnp.array([lx, 0.0]),
                v=jnp.array([0.0, ly]),
            ),
            approximate_num_terms=max(fourier_order[0], fourier_order[1]),
            truncation=basis.Truncation.CIRCULAR,
        )

        pol_runs = self._source.get_polarization_runs()
        all_qe: dict[str, list[float]] = {}
        all_R: list[float] = []
        all_T: list[float] = []
        all_A: list[float] = []

        for _wl_idx, wavelength in enumerate(self._source.wavelengths):
            layer_slices = self._pixel_stack.get_layer_slices(
                wavelength, nx, ny
            )

            R_pol: list[float] = []
            T_pol: list[float] = []
            A_pol: list[float] = []
            qe_pol_accum: dict[str, list[float]] = {}

            for pol in pol_runs:
                try:
                    R, T, A = self._solve_single(
                        wavelength=wavelength,
                        layer_slices=layer_slices,
                        pol=pol,
                        expansion=expansion,
                        formulation=formulation,
                        jax_dtype=jax_dtype,
                        lx=lx,
                        ly=ly,
                        jnp=jnp,
                        fmm=fmm,
                    )
                except Exception as e:
                    logger.error(
                        f"fmmax failed at lambda={wavelength:.4f}um, "
                        f"pol={pol}: {e}"
                    )
                    R, T, A = 0.0, 0.0, 0.0

                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)

                self._last_layer_slices = layer_slices
                self._last_wavelength = wavelength

                pixel_qe = self._compute_per_pixel_qe(
                    layer_slices, wavelength, A
                )
                for key, val in pixel_qe.items():
                    qe_pol_accum.setdefault(key, []).append(val)

            # Average over polarization runs
            n_pol = len(pol_runs)
            all_R.append(sum(R_pol) / n_pol)
            all_T.append(sum(T_pol) / n_pol)
            all_A.append(sum(A_pol) / n_pol)

            for k, vals in qe_pol_accum.items():
                all_qe.setdefault(k, []).append(sum(vals) / n_pol)

        # Build result arrays and check for NaN/Inf
        result_arrays = {
            "reflection": np.array(all_R),
            "transmission": np.array(all_T),
            "absorption": np.array(all_A),
        }
        for arr_name, arr in result_arrays.items():
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                warnings.warn(
                    f"fmmax: NaN/Inf detected in {arr_name} output",
                    stacklevel=2,
                )

        return SimulationResult(
            qe_per_pixel={k: np.array(v) for k, v in all_qe.items()},
            wavelengths=self._source.wavelengths,
            **result_arrays,
            metadata={
                "solver_name": "fmmax",
                "fmm_formulation": self._fmm_formulation,
                "fourier_order": fourier_order,
                "dtype": self._dtype_str,
            },
        )

    def _solve_single(
        self,
        wavelength: float,
        layer_slices: list,
        pol: str,
        expansion,
        formulation,
        jax_dtype,
        lx: float,
        ly: float,
        jnp,
        fmm,
    ) -> tuple[float, float, float]:
        """Run fmmax for a single wavelength and polarization.

        Args:
            wavelength: Wavelength in um.
            layer_slices: List of LayerSlice from pixel_stack.
            pol: Polarization string ("TE" or "TM").
            expansion: fmmax basis expansion.
            formulation: fmmax FMM formulation enum.
            jax_dtype: JAX complex dtype.
            lx: Domain size in x (um).
            ly: Domain size in y (um).
            jnp: jax.numpy module reference.
            fmm: fmmax.fmm module reference.

        Returns:
            Tuple of (R, T, A) as floats.
        """
        # Convert wavelength from um to the unit used by fmmax
        # fmmax works in user-defined length units; we use um consistently
        in_plane_wavevector = jnp.zeros(2, dtype=float)

        # Apply incidence angles if non-zero
        if self._source is not None:
            theta_rad = self._source.theta_rad
            phi_rad = self._source.phi_rad
            if theta_rad != 0.0 or phi_rad != 0.0:
                k0 = 2.0 * np.pi / wavelength
                kx = k0 * np.sin(theta_rad) * np.cos(phi_rad)
                ky = k0 * np.sin(theta_rad) * np.sin(phi_rad)
                in_plane_wavevector = jnp.array([kx, ky])

        # Set polarization amplitudes
        if pol == "TE":
            # s-polarization: E perpendicular to plane of incidence
            s_amplitude = jnp.ones(1, dtype=jax_dtype)
            p_amplitude = jnp.zeros(1, dtype=jax_dtype)
        else:
            # p-polarization: E in plane of incidence
            s_amplitude = jnp.zeros(1, dtype=jax_dtype)
            p_amplitude = jnp.ones(1, dtype=jax_dtype)

        # Build layer permittivities and thicknesses
        # fmmax layers are ordered from top (superstrate side) to bottom (substrate side)
        # COMPASS layer_slices are bottom-to-top, so we reverse
        layer_thicknesses = []
        layer_permittivities = []

        for s in reversed(layer_slices):
            layer_thicknesses.append(s.thickness)
            # Convert numpy permittivity grid to JAX array
            eps_jax = jnp.array(s.eps_grid, dtype=jax_dtype)
            layer_permittivities.append(eps_jax)

        # Solve each layer eigenmode problem
        layer_solve_results = []
        for eps in layer_permittivities:
            solve_result = fmm.eigensolve_isotropic_media(
                wavelength=jnp.array(wavelength),
                in_plane_wavevector=in_plane_wavevector,
                primitive_lattice_vectors=expansion.primitive_lattice_vectors,
                permittivity=eps,
                expansion=expansion,
                formulation=formulation,
            )
            layer_solve_results.append(solve_result)

        # Build S-matrix by cascading layers
        s_matrix = layer_solve_results[0].s_matrix
        for i in range(1, len(layer_solve_results)):
            s_matrix = fmm.append_layer(
                s_matrix=s_matrix,
                layer_solve_result=layer_solve_results[i],
                layer_thickness=jnp.array(layer_thicknesses[i]),
            )

        # Compute R and T power from S-matrix
        # Total reflected and transmitted power
        r_coeffs = s_matrix.s11 @ s_amplitude + s_matrix.s12 @ p_amplitude
        t_coeffs = s_matrix.s21 @ s_amplitude + s_matrix.s22 @ p_amplitude

        R = float(jnp.sum(jnp.abs(r_coeffs) ** 2))
        T = float(jnp.sum(jnp.abs(t_coeffs) ** 2))
        A = max(0.0, 1.0 - R - T)

        return R, T, A

    def _compute_per_pixel_qe(
        self,
        layer_slices: list,
        wavelength: float,
        total_absorption: float,
    ) -> dict[str, float]:
        """Compute per-pixel QE using eps_imag weighting in PD regions.

        Args:
            layer_slices: Layer slices from pixel_stack.
            wavelength: Current wavelength in um.
            total_absorption: Total absorption for this wavelength.

        Returns:
            Dictionary mapping pixel key to QE value.
        """
        if self._pixel_stack is None:
            raise RuntimeError(
                "pixel_stack is not set; call setup_geometry() first"
            )
        bayer = self._pixel_stack.bayer_map
        n_rows, n_cols = self._pixel_stack.unit_cell
        n_pixels = n_rows * n_cols
        if n_pixels == 0:
            return {}
        lx, ly = self._pixel_stack.domain_size

        pixel_weights: dict[str, float] = {}
        total_weight = 0.0

        for pd in self._pixel_stack.photodiodes:
            r, c = pd.pixel_index
            key = f"{pd.color}_{r}_{c}"
            pd_z_min = pd.position[2] - pd.size[2] / 2
            pd_z_max = pd.position[2] + pd.size[2] / 2

            weight = 0.0
            for s in layer_slices:
                z_overlap_min = max(s.z_start, pd_z_min)
                z_overlap_max = min(s.z_end, pd_z_max)
                if z_overlap_max <= z_overlap_min:
                    continue
                dz = z_overlap_max - z_overlap_min
                nx_s, ny_s = s.eps_grid.shape
                ix_min = max(
                    0,
                    int(
                        ((pd.position[0] - pd.size[0] / 2 + lx / 2) / lx)
                        * nx_s
                    ),
                )
                ix_max = min(
                    nx_s,
                    int(
                        ((pd.position[0] + pd.size[0] / 2 + lx / 2) / lx)
                        * nx_s
                    ),
                )
                iy_min = max(
                    0,
                    int(
                        ((pd.position[1] - pd.size[1] / 2 + ly / 2) / ly)
                        * ny_s
                    ),
                )
                iy_max = min(
                    ny_s,
                    int(
                        ((pd.position[1] + pd.size[1] / 2 + ly / 2) / ly)
                        * ny_s
                    ),
                )
                if ix_max <= ix_min or iy_max <= iy_min:
                    continue
                eps_region = s.eps_grid[ix_min:ix_max, iy_min:iy_max]
                weight += float(np.mean(np.imag(eps_region))) * dz

            pixel_weights[key] = max(weight, 0.0)
            total_weight += max(weight, 0.0)

        qe_per_pixel: dict[str, float] = {}
        if total_weight > 0:
            for key, w in pixel_weights.items():
                qe_per_pixel[key] = total_absorption * (w / total_weight)
        else:
            for r in range(n_rows):
                for c in range(n_cols):
                    key = f"{bayer[r][c]}_{r}_{c}"
                    qe_per_pixel[key] = total_absorption / n_pixels
        return qe_per_pixel

    def get_field_distribution(
        self,
        component: str = "|E|2",
        plane: str = "xz",
        position: float = 0.0,
    ) -> np.ndarray:
        """Extract approximate 2D field from permittivity profile.

        This provides a Beer-Lambert approximation of the field distribution
        based on the imaginary part of permittivity, as full field extraction
        from fmmax requires additional computation.

        Args:
            component: Field component ("Ex", "Ey", "Ez", "|E|2").
            plane: Slice plane ("xy", "xz", "yz").
            position: Position along the normal axis in um.

        Returns:
            2D field array.
        """
        if self._last_layer_slices is None:
            logger.warning("fmmax: no simulation data, returning zeros")
            return np.zeros((64, 64))

        if self._pixel_stack is None:
            raise RuntimeError(
                "pixel_stack is not set; call setup_geometry() first"
            )
        layer_info = self._last_layer_slices
        _lx, ly = self._pixel_stack.domain_size
        nz = len(layer_info)
        nx_out, ny_out = 64, 64
        wl = self._last_wavelength or 0.55
        k0 = 2 * np.pi / wl

        if plane == "xz":
            field_2d = np.zeros((nx_out, nz))
            for zi, s in enumerate(layer_info):
                eps = s.eps_grid
                ny_s = eps.shape[1]
                y_idx = max(
                    0,
                    min(ny_s - 1, int(((position + ly / 2) / ly) * ny_s)),
                )
                col = eps[:, y_idx]
                x_orig = np.linspace(0, 1, len(col))
                x_new = np.linspace(0, 1, nx_out)
                field_2d[:, zi] = np.interp(
                    x_new, x_orig, np.abs(np.imag(col)) + 1e-10
                )
            if component == "|E|2":
                for xi in range(nx_out):
                    intensity = 1.0
                    for zi in range(nz):
                        alpha = 2 * k0 * field_2d[xi, zi]
                        dz = layer_info[zi].thickness
                        intensity *= np.exp(-alpha * dz)
                        field_2d[xi, zi] = intensity
            return field_2d

        elif plane == "xy":
            z_accum = 0.0
            for s in layer_info:
                if z_accum + s.thickness >= position:
                    from scipy.ndimage import zoom

                    eps = s.eps_grid
                    zx = nx_out / eps.shape[0]
                    zy = ny_out / eps.shape[1]
                    return np.asarray(
                        np.abs(zoom(np.real(eps), (zx, zy), order=1))
                    )
                z_accum += s.thickness

        return np.zeros((64, 64))


SolverFactory.register("fmmax", FmmaxSolver)
