"""fmmax (Fourier Modal Method with JAX) RCWA solver adapter for COMPASS.

Wraps the fmmax library, which implements the Fourier Modal Method (FMM)
using JAX for automatic differentiation and GPU acceleration.
Supports multiple FMM formulations: Pol, Normal, Jones, JonesDirect.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from compass.compute import cuda_device_index
from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.solvers.rcwa.fields import (
    FIELD_N_XY,
    approximate_field,
    check_xy_position,
    column_heights,
    component_from_field,
    eps_imag_weight_qe,
    integrate_photodiode_qe,
    locate_slice,
    slice_depth_samples,
    validate_field_request,
)
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)

# Valid FMM formulation names mapping to fmmax enums
_FORMULATION_MAP = {
    "pol": "POL",
    "normal": "NORMAL",
    "jones": "JONES",
    "jonesdirect": "JONES_DIRECT",
}


class _FmmaxFieldSampler:
    """Field evaluation inside a solved fmmax layer stack.

    Interior amplitudes come from fmmax's ``stack_s_matrices_interior`` /
    ``stack_amplitudes_interior``; fields are evaluated with a direct
    Fourier sum at arbitrary points, batched over depth.

    Conventions, validated against the flux and Poynting-theorem invariants:

    * fmmax z points down from each layer's top (like torcwa/grcwa) and
      uses ``exp(-i omega t)``; layer 0 / N+1 are the semi-infinite air
      media, so slice ``i`` is fmmax layer ``i + 1``.
    * The incident amplitude is rescaled so ``Re(E x H*)`` integrates to
      ``cos(theta)`` over a z-plane (fmmax reports ``0.5 Re(E x H*)``),
      i.e. |E0| = 1 as in the other RCWA adapters.
    * fmmax samples the permittivity at cell centres, matching COMPASS.
    * The vector formulations factorize the permittivity with normal-vector
      (Li) rules, under which the truncated E series rings at metal /
      high-contrast interfaces. The real-space E is reconstructed from the
      displacement, ``E = D / eps(r)``, using fmmax's own transverse and zz
      permittivity matrices.
    """

    def __init__(self, layer_solve_results, layer_thicknesses, forward, power,
                 cos_theta, eps_grids, lx, ly):
        import jax.numpy as jnp
        from fmmax import basis, fields, scattering

        self._jnp = jnp
        self._fields = fields
        self.lsr = layer_solve_results
        self.thick = [float(t) for t in layer_thicknesses]
        self.eps_grids = eps_grids  # COMPASS [y, x] grids, slice order (top first)
        self.lx, self.ly = lx, ly
        scale = float(np.sqrt(cos_theta / (2.0 * power)))
        fwd = forward * scale
        interior = scattering.stack_s_matrices_interior(
            layer_solve_results, [jnp.asarray(t) for t in layer_thicknesses]
        )
        self.amps = fields.stack_amplitudes_interior(interior, fwd, jnp.zeros_like(fwd))
        lsr0 = layer_solve_results[0]
        self.k = np.asarray(basis.transverse_wavevectors(
            in_plane_wavevector=lsr0.in_plane_wavevector,
            primitive_lattice_vectors=lsr0.primitive_lattice_vectors,
            expansion=lsr0.expansion,
        ))  # (n_terms, 2)

    def fourier(self, slice_index: int, depth_um):
        """Fourier coefficients ([ex, ey, ez], [hx, hy, hz], [dx, dy, dz]), each (n_depth, N)."""
        jnp = self._jnp
        L = slice_index + 1
        lsr = self.lsr[L]
        z = jnp.asarray(np.atleast_1d(np.asarray(depth_um, dtype=float)))
        fwd_start, bwd_end = self.amps[L]
        fwd, bwd = self._fields.colocate_amplitudes(
            fwd_start, bwd_end, z_offset=z, layer_solve_result=lsr,
            layer_thickness=jnp.asarray(self.thick[L]),
        )
        (ex, ey, ez), (hx, hy, hz) = self._fields.fields_from_wave_amplitudes(fwd, bwd, lsr)
        n = ex.shape[-2]
        d_t = lsr.transverse_permittivity_matrix @ jnp.concatenate([ex, ey], axis=-2)
        dz = lsr.z_permittivity_matrix @ ez

        def np_t(a):
            return np.asarray(a).T

        return (
            [np_t(ex), np_t(ey), np_t(ez)],
            [np_t(hx), np_t(hy), np_t(hz)],
            [np_t(d_t[..., :n, :]), np_t(d_t[..., n:, :]), np_t(dz)],
        )

    def power_down(self, slice_index: int, depth_um: float) -> float:
        """Cell-averaged downward Re(E x H*) (incident flux = cos(theta))."""
        (ex, ey, _), (hx, hy, _), _ = self.fourier(slice_index, np.array([depth_um]))
        return float(np.sum(np.real(ex * np.conj(hy) - ey * np.conj(hx))))

    def _eps_at(self, slice_index: int, x_um, y_um) -> np.ndarray:
        eps = self.eps_grids[slice_index]
        n0, n1 = eps.shape
        ix = np.floor(np.asarray(x_um) / self.lx * n1).astype(int) % n1
        iy = np.floor(np.asarray(y_um) / self.ly * n0).astype(int) % n0
        return np.asarray(eps[iy, ix])

    def _phase(self, x_um, y_um) -> np.ndarray:
        x = np.asarray(x_um, dtype=float)
        y = np.asarray(y_um, dtype=float)
        kx = self.k[:, 0].reshape((-1,) + (1,) * x.ndim)
        ky = self.k[:, 1].reshape((-1,) + (1,) * x.ndim)
        return np.asarray(np.exp(1j * (kx * x + ky * y)))  # (N, *points)

    def e2_on_grid(self, slice_index: int, depths_um, x_um, y_um) -> np.ndarray:
        """Sum over depths of |E|^2 = |D/eps|^2 on the grid, indexed [y, x]."""
        _, _, fd = self.fourier(slice_index, depths_um)
        Y, X = np.meshgrid(np.asarray(y_um), np.asarray(x_um), indexing="ij")
        phase = self._phase(X, Y).reshape(len(self.k), -1)  # (N, n_pts)
        total = np.zeros(phase.shape[1])
        for c in fd:
            total += np.sum(np.abs(c @ phase) ** 2, axis=0)
        inv_eps2 = 1.0 / np.abs(self._eps_at(slice_index, X, Y)) ** 2
        return np.asarray(total.reshape(X.shape) * inv_eps2)

    def at_points(
        self, slice_index: int, depth_um: float, x_um, y_um, displacement: bool = True
    ):
        """Fields ([Ex, Ey, Ez], [Hx, Hy, Hz]) at points (x, y) in um.

        With ``displacement`` (default) E is ``D / eps(r)``; otherwise the
        raw truncated E series, whose plane average of ``Re(E x H*)`` equals
        the Fourier-space flux exactly (used for ``Sz``).
        """
        fe, fh, fd = self.fourier(slice_index, np.array([depth_um]))
        phase = self._phase(x_um, y_um)
        shape = (-1,) + (1,) * (phase.ndim - 1)

        def series(c):
            return np.sum(c[0].reshape(shape) * phase, axis=0)

        H = [series(c) for c in fh]
        if not displacement:
            return [series(c) for c in fe], H
        eps = self._eps_at(slice_index, x_um, y_um)
        return [series(c) / eps for c in fd], H


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
        self._last_layer_info: list | None = None
        self._last_sampler: _FmmaxFieldSampler | None = None
        self._last_wavelength: float | None = None
        self._qe_fallback_used = False

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
        logger.info(f"fmmax: geometry setup, formulation={self._fmm_formulation}")

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
        logger.info(f"fmmax: source setup - {self._source.n_wavelengths} wavelengths")

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
                "fmmax and jax are required. Install with: pip install fmmax jax"
            ) from err

        # Configure JAX device
        target = "gpu" if self.device.startswith("cuda") else "cpu"
        selected = jax.devices(target)[cuda_device_index(self.device)]
        jax.config.update("jax_default_device", selected)
        if self._dtype_str == "complex128" and not jax.config.read("jax_enable_x64"):
            # Without x64 JAX silently downcasts complex128 to complex64.
            jax.config.update("jax_enable_x64", True)

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
        if self._dtype_str not in dtype_map:
            raise ValueError(
                f"fmmax dtype must be one of {sorted(dtype_map)}, got {self._dtype_str!r}"
            )
        jax_dtype = dtype_map[self._dtype_str]

        lx, ly = self._pixel_stack.domain_size
        # Zero disables automatic oversampling for independent grid/order studies.
        grid_multiplier = params.get("grid_multiplier", 3)
        nx = max(int(params.get("epsilon_grid", 64)), (2 * fourier_order[0] + 1) * grid_multiplier)
        ny = max(int(params.get("epsilon_grid", 64)), (2 * fourier_order[1] + 1) * grid_multiplier)

        # Match the per-axis order convention of torcwa and meent exactly.
        # generate_expansion's approximate_num_terms is a TOTAL count.
        mx, my = fourier_order
        coefficients = np.array([
            (i, j) for i in range(-mx, mx + 1) for j in range(-my, my + 1)
        ])
        coefficients = coefficients[np.argsort(
            np.sum(coefficients ** 2, axis=1), kind="stable",
        )]
        expansion = basis.Expansion(basis_coefficients=coefficients)

        pol_runs = self._source.get_polarization_runs()
        self._failed_runs = []
        self._qe_fallback_used = False
        all_qe: dict[str, list[float]] = {}
        all_R: list[float] = []
        all_T: list[float] = []
        all_A: list[float] = []

        for _wl_idx, wavelength in enumerate(self._source.wavelengths):
            layer_slices = self._pixel_stack.get_layer_slices(
                wavelength, nx, ny, n_lens_slices=params.get("n_lens_slices", 30),
            )

            R_pol: list[float] = []
            T_pol: list[float] = []
            A_pol: list[float] = []
            qe_pol_accum: dict[str, list[float]] = {}

            for pol in pol_runs:
                sampler: _FmmaxFieldSampler | None = None
                try:
                    R, T, A, sampler = self._solve_single(
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
                    logger.error(f"fmmax failed at lambda={wavelength:.4f}um, pol={pol}: {e}")
                    self._record_failed_run(wavelength, pol, e)
                    R = T = A = float("nan")

                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)

                self._last_layer_slices = layer_slices
                self._last_layer_info = list(reversed(layer_slices))
                self._last_wavelength = wavelength
                self._last_sampler = sampler

                if np.isnan(A):
                    pixel_qe = self._nan_pixel_qe()
                elif not self._pixel_stack.photodiodes:
                    pixel_qe = {}
                else:
                    pixel_qe = self._compute_qe_with_fallback(sampler, layer_slices, wavelength, A)
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
            reflection=result_arrays["reflection"],
            transmission=result_arrays["transmission"],
            absorption=result_arrays["absorption"],
            metadata={
                "solver_name": "fmmax",
                "fmm_formulation": self._fmm_formulation,
                "fourier_order": fourier_order,
                "epsilon_grid_requested": int(params.get("epsilon_grid", 64)),
                "epsilon_grid_shape_yx": [ny, nx],
                "grid_multiplier": grid_multiplier,
                "num_terms": expansion.num_terms,
                "dtype": self._dtype_str,
                # Field integration vs. the eps''-weight approximation is a
                # QE methodology difference, not solver accuracy.
                "qe_method": (
                    "eps_imag_weight_fallback" if self._qe_fallback_used else "field_integration"
                ),
                **self._failure_metadata(),
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
    ) -> tuple[float, float, float, _FmmaxFieldSampler | None]:
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
            Tuple of (R, T, A) as floats and a field sampler (None if the
            interior field setup failed).
        """
        from fmmax import basis, fields, scattering

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

        lattice = basis.LatticeVectors(
            u=jnp.array([lx, 0.0]), v=jnp.array([0.0, ly]),
        )

        # Build layer permittivities and thicknesses
        # fmmax layers are ordered from top (superstrate side) to bottom (substrate side)
        # COMPASS layer_slices are bottom-to-top, so we reverse
        layer_thicknesses = [0.0]
        layer_permittivities = [jnp.ones((1, 1), dtype=jax_dtype)]

        for s in reversed(layer_slices):
            layer_thicknesses.append(s.thickness)
            # Convert numpy permittivity grid to JAX array
            # fmmax uses [x, y], while COMPASS stores grids as [y, x].
            eps_jax = jnp.array(s.eps_grid.T, dtype=jax_dtype)
            layer_permittivities.append(eps_jax)

        layer_thicknesses.append(0.0)
        layer_permittivities.append(jnp.ones((1, 1), dtype=jax_dtype))

        # Solve each layer eigenmode problem
        layer_solve_results = []
        for eps in layer_permittivities:
            solve_result = fmm.eigensolve_isotropic_media(
                wavelength=jnp.array(wavelength),
                in_plane_wavevector=in_plane_wavevector,
                primitive_lattice_vectors=lattice,
                permittivity=eps,
                expansion=expansion,
                formulation=formulation,
            )
            layer_solve_results.append(solve_result)

        s_matrix = scattering.stack_s_matrix(
            layer_solve_results, [jnp.array(d) for d in layer_thicknesses],
        )

        # The homogeneous air eigenvectors span Hx and Hy for each order.
        # Excite only the zero diffraction order, rotated into the s/p basis.
        n_terms = expansion.num_terms
        zero_order = int(np.flatnonzero(
            np.all(np.asarray(expansion.basis_coefficients) == 0, axis=1)
        )[0])
        phi = self._source.phi_rad if self._source is not None else 0.0
        incident = np.zeros((2 * n_terms, 1), dtype=complex)
        hx, hy = (np.cos(phi), np.sin(phi)) if pol == "TE" else (-np.sin(phi), np.cos(phi))
        incident[zero_order, 0] = hx
        incident[n_terms + zero_order, 0] = hy
        forward = jnp.array(incident, dtype=jax_dtype)
        zero = jnp.zeros(forward.shape, dtype=jax_dtype)
        reflected = s_matrix.s21 @ forward
        transmitted = s_matrix.s11 @ forward
        incident_flux, reflected_flux = fields.amplitude_poynting_flux(
            forward, reflected, layer_solve_results[0],
        )
        transmitted_flux, _ = fields.amplitude_poynting_flux(
            transmitted, zero, layer_solve_results[-1],
        )
        power = float(jnp.sum(incident_flux))
        if not np.isfinite(power) or power <= 0:
            raise RuntimeError("fmmax incident power must be finite and positive")
        R = -float(jnp.sum(reflected_flux)) / power
        T = float(jnp.sum(transmitted_flux)) / power
        A = 1.0 - R - T

        try:
            cos_theta = float(np.cos(self._source.theta_rad)) if self._source else 1.0
            sampler: _FmmaxFieldSampler | None = _FmmaxFieldSampler(
                layer_solve_results, layer_thicknesses, forward, power, cos_theta,
                [s.eps_grid for s in reversed(layer_slices)], lx, ly,
            )
        except Exception as e:  # field data unavailable -> QE fallback
            logger.warning(f"fmmax: field reconstruction setup failed ({e})")
            sampler = None
        return R, T, A, sampler

    def _compute_qe_with_fallback(
        self,
        sampler: _FmmaxFieldSampler | None,
        layer_slices: list,
        wavelength: float,
        total_absorption: float,
    ) -> dict[str, float]:
        """Field-integrated QE, falling back to eps''-weighting on failure."""
        try:
            if sampler is None:
                raise RuntimeError("no field data")
            return self._compute_per_pixel_qe_fields(sampler, wavelength, total_absorption)
        except Exception as e:
            logger.warning(
                f"fmmax: field-based QE failed at λ={wavelength:.4f}um ({e}); "
                "falling back to eps''-weight approximation"
            )
            self._qe_fallback_used = True
            return self._compute_per_pixel_qe(layer_slices, wavelength, total_absorption)

    def _compute_per_pixel_qe_fields(
        self,
        sampler: _FmmaxFieldSampler,
        wavelength: float,
        total_absorption: float,
    ) -> dict[str, float]:
        """Per-pixel QE by integrating eps'' |E|^2 over photodiode volumes.

        Uses the QE definition shared with the other RCWA adapters
        (:func:`compass.solvers.rcwa.fields.integrate_photodiode_qe`); each
        slice's total is pinned to the exact Fourier-space flux difference.
        """
        if self._pixel_stack is None or self._source is None:
            raise RuntimeError("Call setup_geometry() and setup_source() before run()")
        if self._last_layer_info is None:
            raise RuntimeError("no layer data recorded")
        lx, ly = self._pixel_stack.domain_size
        cos_theta = float(np.cos(self._source.theta_rad))

        def e2_dz(li: int, s) -> np.ndarray:
            n0, n1 = s.eps_grid.shape  # [y, x]
            depths, dz = slice_depth_samples(s.thickness)
            x = (np.arange(n1) + 0.5) * (lx / n1)  # COMPASS cell centres
            y = (np.arange(n0) + 0.5) * (ly / n0)
            return sampler.e2_on_grid(li, depths, x, y) * dz

        def absorbed(li: int, s) -> float:
            top = sampler.power_down(li, 0.0)
            return (top - sampler.power_down(li, s.thickness)) / cos_theta

        return integrate_photodiode_qe(
            self._pixel_stack,
            self._last_layer_info,
            wavelength,
            self._source.theta_rad,
            e2_dz,
            total_absorption,
            "fmmax",
            slice_absorption=absorbed,
        )

    def _compute_per_pixel_qe(
        self,
        layer_slices: list,
        wavelength: float,
        total_absorption: float,
    ) -> dict[str, float]:
        """Approximate per-pixel QE by eps'' weighting (fallback only)."""
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        return eps_imag_weight_qe(self._pixel_stack, layer_slices, total_absorption)

    def get_field_distribution(
        self,
        component: str = "|E|2",
        plane: str = "xz",
        position: float = 0.0,
    ) -> np.ndarray:
        """Extract a 2D solved-field slice from the last simulation.

        Same conventions as the other RCWA adapters: x, y in
        ``[0, lx) x [0, ly)``, z up from the stack bottom, |E0| = 1, and
        ``Sz`` the Poynting z-component (negative for downward flow;
        incident flux ``cos(theta)``). Fields belong to the last
        polarization run. Intensities use the displacement-based
        reconstruction ``E = D / eps``; ``Sz`` uses the raw E series so its
        plane average is the exact Fourier-space flux.

        Args:
            component: "Ex", "Ey", "Ez" (each |E_c|^2), "|E|2", or "Sz".
            plane: "xy" (position = z), "xz" (position = y) or
                "yz" (position = x), all in um.
            position: Position along the normal axis in um.

        Returns:
            2D array: ``(nx, ny)`` for "xy", ``(nx, nz)`` for "xz",
            ``(ny, nz)`` for "yz", with z increasing upward.

        Raises:
            ValueError: Unknown component/plane, or an "xy" position outside
                the simulated stack.
        """
        validate_field_request(component, plane)
        if self._last_layer_info is None or self._pixel_stack is None:
            logger.warning("fmmax: no simulation data, returning zeros")
            return np.zeros((FIELD_N_XY, FIELD_N_XY))
        if plane == "xy":
            check_xy_position(self._last_layer_info, position)
        try:
            if self._last_sampler is None:
                raise RuntimeError("no field data from the last run")
            return self._solved_field(self._last_sampler, component, plane, position)
        except Exception as e:
            logger.warning(
                f"fmmax: field reconstruction failed ({e}); returning a "
                "permittivity-based approximation, not a solved field"
            )
        return approximate_field(
            self._pixel_stack, self._last_layer_info, self._last_wavelength,
            component, plane, position,
        )

    def _solved_field(
        self, sampler: _FmmaxFieldSampler, component: str, plane: str, position: float
    ) -> np.ndarray:
        if self._pixel_stack is None or self._last_layer_info is None:
            raise RuntimeError("no simulation data")
        layer_info = self._last_layer_info
        lx, ly = self._pixel_stack.domain_size
        u = np.arange(FIELD_N_XY) / FIELD_N_XY

        if plane == "xy":
            li, depth = locate_slice(layer_info, position)
            X, Y = np.meshgrid(u * lx, u * ly, indexing="ij")
            E, H = sampler.at_points(li, depth, X, Y, displacement=component != "Sz")
            return component_from_field(E, H, component)

        heights = column_heights(layer_info)
        line = u * (lx if plane == "xz" else ly)
        x = line if plane == "xz" else np.full_like(line, position)
        y = np.full_like(line, position) if plane == "xz" else line
        out = np.zeros((FIELD_N_XY, len(heights)))
        for zi, z in enumerate(heights):
            li, depth = locate_slice(layer_info, float(z))
            E, H = sampler.at_points(li, depth, x, y, displacement=component != "Sz")
            out[:, zi] = component_from_field(E, H, component)
        return out


SolverFactory.register("fmmax", FmmaxSolver)
