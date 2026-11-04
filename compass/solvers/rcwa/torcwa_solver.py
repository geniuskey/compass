"""torcwa RCWA solver adapter for COMPASS.

Wraps the torcwa library to implement the SolverBase interface.
torcwa is a PyTorch-based RCWA solver with GPU support.
"""

from __future__ import annotations

import logging

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.solvers.rcwa.fields import (
    FIELD_COMPONENTS,
    FIELD_N_XY,
    FIELD_N_Z,
    FIELD_PLANES,
    approximate_field,
    check_xy_position,
    column_heights,
    component_from_field,
    eps_imag_weight_qe,
    integrate_photodiode_qe,
    locate_slice,
    slice_depth_samples,
    stack_z_range,
    validate_field_request,
)
from compass.sources.planewave import PlanewaveSource

logger = logging.getLogger(__name__)


class TorcwaSolver(SolverBase):
    """torcwa RCWA solver adapter.

    Converts PixelStack geometry to torcwa layer structure and executes
    RCWA simulation with S-matrix algorithm.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._layer_slices = None
        self._sim = None
        self._last_sim = None
        self._last_layer_info: list | None = None
        self._last_wavelength: float | None = None
        self._last_source_amp: list[float] | None = None
        self._qe_fallback_used = False

        # Configure precision
        self._setup_precision()

    def _setup_precision(self) -> None:
        """Configure numerical precision settings (TF32 off on CUDA)."""
        from compass.solvers.rcwa.stability import PrecisionManager

        PrecisionManager.configure(self.config)

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        """Convert PixelStack to torcwa layer structure."""
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")
        self._pixel_stack = pixel_stack
        logger.info(
            f"torcwa: geometry setup for {pixel_stack.unit_cell} unit cell, "
            f"pitch={pixel_stack.pitch}um"
        )

    def setup_source(self, source_config: dict) -> None:
        """Configure planewave source from config."""
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config
        logger.info(
            f"torcwa: source setup - {self._source.n_wavelengths} wavelengths, "
            f"theta={self._source.theta_deg}deg, pol={self._source.polarization}"
        )

    def run(self) -> SimulationResult:
        """Execute RCWA simulation for all wavelengths."""
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        try:
            import torch
            import torcwa
        except ImportError as err:
            raise ImportError("torcwa is required. Install with: pip install torcwa") from err

        params = self.config.get("params", {})
        fourier_order = params.get("fourier_order", [9, 9])
        dtype_str = params.get("dtype", "complex64")
        dtype = getattr(torch, dtype_str)
        n_lens_slices = params.get("n_lens_slices", 30)
        grid_multiplier = params.get("grid_multiplier", 3)

        lx, ly = self._pixel_stack.domain_size
        L = [lx, ly]  # Period in um

        nx = max(int(params.get("epsilon_grid", 64)), (2 * fourier_order[0] + 1) * grid_multiplier)
        ny = max(int(params.get("epsilon_grid", 64)), (2 * fourier_order[1] + 1) * grid_multiplier)

        pol_runs = self._source.get_polarization_runs()
        all_qe: dict[str, list[np.ndarray]] = {}
        all_R, all_T, all_A = [], [], []
        self._failed_runs = []
        self._qe_fallback_used = False

        for wl_idx, wavelength in enumerate(self._source.wavelengths):
            logger.debug(
                f"torcwa: wavelength {wavelength:.4f} um ({wl_idx + 1}/{self._source.n_wavelengths})"
            )

            layer_slices = self._pixel_stack.get_layer_slices(
                wavelength,
                nx,
                ny,
                n_lens_slices=n_lens_slices,
            )

            qe_pol_accum: dict[str, list] = {}
            R_pol, T_pol, A_pol = [], [], []

            for pol in pol_runs:
                try:
                    result = self._run_single(
                        torcwa,
                        torch,
                        wavelength,
                        L,
                        fourier_order,
                        layer_slices,
                        pol,
                        dtype,
                    )
                    R_pol.append(result["R"])
                    T_pol.append(result["T"])
                    A_pol.append(result["A"])

                    for k, v in result["qe_per_pixel"].items():
                        qe_pol_accum.setdefault(k, []).append(v)

                except Exception as e:
                    logger.error(f"torcwa: failed at λ={wavelength:.4f}um, pol={pol}: {e}")
                    self._record_failed_run(wavelength, pol, e)
                    R_pol.append(float("nan"))
                    T_pol.append(float("nan"))
                    A_pol.append(float("nan"))
                    for k, v in self._nan_pixel_qe().items():
                        qe_pol_accum.setdefault(k, []).append(v)

            # Average over polarizations
            n_pol = len(pol_runs)
            all_R.append(sum(R_pol) / n_pol)
            all_T.append(sum(T_pol) / n_pol)
            all_A.append(sum(A_pol) / n_pol)

            for k, vals in qe_pol_accum.items():
                all_qe.setdefault(k, []).append(sum(vals) / n_pol)

        # Assemble result
        qe_per_pixel = {k: np.array(v) for k, v in all_qe.items()}

        result_arrays = {
            "reflection": np.array(all_R),
            "transmission": np.array(all_T),
            "absorption": np.array(all_A),
        }
        for arr_name, arr in result_arrays.items():
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                import warnings

                warnings.warn(f"torcwa: NaN/Inf detected in {arr_name} output", stacklevel=2)

        return SimulationResult(
            qe_per_pixel=qe_per_pixel,
            wavelengths=self._source.wavelengths,
            reflection=result_arrays["reflection"],
            transmission=result_arrays["transmission"],
            absorption=result_arrays["absorption"],
            metadata={
                "solver_name": "torcwa",
                "fourier_order": fourier_order,
                # QE post-processing method actually used: cross-solver QE
                # differences between field integration and the eps''-weight
                # approximation are methodology, not solver accuracy.
                "qe_method": (
                    "eps_imag_weight_fallback"
                    if self._qe_fallback_used
                    else "field_integration"
                ),
                "device": self.device,
                **self._failure_metadata(),
            },
        )

    def _run_single(
        self,
        torcwa,
        torch,
        wavelength: float,
        L: list,
        fourier_order: list,
        layer_slices,
        polarization: str,
        dtype,
    ) -> dict:
        """Run single wavelength, single polarization RCWA calculation."""
        freq = 1.0 / wavelength  # torcwa uses normalized frequency

        sim = torcwa.rcwa(
            freq=freq,
            order=fourier_order,
            L=L,
            dtype=dtype,
            device=self.device,
        )

        # Input layer (air, above structure)
        sim.add_input_layer(eps=1.0)

        # Set incidence angle BEFORE adding layers (torcwa needs Kx_norm for eigendecomp)
        if self._source is None:
            raise RuntimeError("source is not set; call setup_source() first")
        sim.set_incident_angle(
            inc_ang=self._source.theta_rad,
            azi_ang=self._source.phi_rad,
        )

        # Track layers for per-pixel QE and field extraction
        layer_info = []
        for s in reversed(layer_slices):
            # LayerSlice is [y, x]; torcwa's Fourier convolution is [x, y].
            eps_tensor = torch.tensor(s.eps_grid.T.copy(), dtype=dtype, device=self.device)
            sim.add_layer(thickness=s.thickness, eps=eps_tensor)
            layer_info.append(s)

        # Solve (no add_output_layer — defaults to free space)
        sim.solve_global_smatrix()

        # Map COMPASS polarization runs onto torcwa's ps notation:
        # TE = s-polarized, TM = p-polarized. amplitude is [Ep, Es].
        if polarization in ("TM", "p", "P"):
            in_tag, source_amp = "p", [1.0, 0.0]
        else:
            in_tag, source_amp = "s", [0.0, 1.0]

        # Extract R, T via S_parameters method
        if callable(sim.S_parameters):
            # Total reflectance/transmittance: sum diffraction efficiency over
            # ALL Fourier orders and BOTH output polarizations. The zeroth
            # order alone underestimates R/T (and overstates A) as soon as
            # higher orders propagate, which is the norm for multi-um Bayer
            # unit cells in the visible.
            all_orders = [
                [m, n]
                for m in range(-fourier_order[0], fourier_order[0] + 1)
                for n in range(-fourier_order[1], fourier_order[1] + 1)
            ]
            R = 0.0
            T = 0.0
            for out_tag in ("s", "p"):
                pol_tag = out_tag + in_tag
                S_R = sim.S_parameters(
                    orders=all_orders,
                    direction="forward",
                    port="reflection",
                    polarization=pol_tag,
                    power_norm=True,
                )
                S_T = sim.S_parameters(
                    orders=all_orders,
                    direction="forward",
                    port="transmission",
                    polarization=pol_tag,
                    power_norm=True,
                )
                R += float(torch.sum(torch.abs(S_R) ** 2))
                T += float(torch.sum(torch.abs(S_T) ** 2))
        else:
            # Legacy: S_parameters is an object with R, T attributes
            R = float(sim.S_parameters.R.real)
            T = float(sim.S_parameters.T.real)
        A = 1.0 - R - T

        # Store sim reference for field extraction
        self._last_sim = sim
        self._last_layer_info = layer_info
        self._last_wavelength = wavelength
        self._last_source_amp = source_amp

        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if not self._pixel_stack.photodiodes:
            return {"R": R, "T": T, "A": A, "qe_per_pixel": {}}

        # Per-pixel QE: integrate the absorbed power density eps'' |E|^2 in
        # the silicon under each photodiode footprint. Falls back to the
        # geometric eps''-weighting approximation if field reconstruction
        # is unavailable.
        try:
            sim.source_planewave(
                amplitude=source_amp,
                direction="forward",
                notation="ps",
            )
            qe_per_pixel = self._compute_per_pixel_qe_fields(sim, torch, layer_info, wavelength, A)
        except Exception as e:
            logger.warning(
                f"torcwa: field-based QE failed at λ={wavelength:.4f}um ({e}); "
                "falling back to eps''-weight approximation"
            )
            self._qe_fallback_used = True
            qe_per_pixel = self._compute_per_pixel_qe(sim, torch, layer_slices, wavelength, A)

        return {"R": R, "T": T, "A": A, "qe_per_pixel": qe_per_pixel}

    def _compute_per_pixel_qe_fields(
        self,
        sim,
        torch,
        layer_info,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
        """Per-pixel QE from torcwa field reconstruction.

        See :func:`compass.solvers.rcwa.fields.integrate_photodiode_qe` for
        the QE definition shared with the other RCWA adapters.

        Args:
            sim: torcwa rcwa object (solved, with a planewave source set).
            torch: torch module.
            layer_info: LayerSlice list in torcwa layer order (input side first).
            wavelength: Wavelength in um.
            total_absorption: A = 1 - R - T for the energy cross-check.

        Returns:
            Dictionary mapping pixel name to QE value.
        """
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        if self._source is None:
            raise RuntimeError("source is not set; call setup_source() first")

        lx, ly = self._pixel_stack.domain_size

        def e2_dz(li: int, s) -> np.ndarray:
            # Sample at the torcwa lattice positions of the eps grid samples.
            n0, n1 = s.eps_grid.shape
            x_axis = torch.arange(n1, device=self.device, dtype=torch.float64) * (lx / n1)
            y_axis = torch.arange(n0, device=self.device, dtype=torch.float64) * (ly / n0)
            depths, dz = slice_depth_samples(s.thickness)
            e2_sum = torch.zeros((n0, n1), device=self.device, dtype=torch.float32)
            for z_prop in depths:
                E, _H = sim.field_xy(li, x_axis, y_axis, z_prop=float(z_prop))
                e2_sum += sum(torch.abs(c) ** 2 for c in E).T
            return np.asarray(e2_sum.cpu().numpy() * dz)

        return integrate_photodiode_qe(
            self._pixel_stack,
            layer_info,
            wavelength,
            self._source.theta_rad,
            e2_dz,
            total_absorption,
            "torcwa",
        )

    def _compute_per_pixel_qe(
        self,
        sim,
        torch,
        layer_slices,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
        """Approximate per-pixel QE by eps'' weighting (fallback only).

        See :func:`compass.solvers.rcwa.fields.eps_imag_weight_qe`; ``sim``,
        ``torch`` and ``wavelength`` are unused and kept for API stability.
        """
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        return eps_imag_weight_qe(self._pixel_stack, layer_slices, total_absorption)

    FIELD_COMPONENTS = FIELD_COMPONENTS
    FIELD_PLANES = FIELD_PLANES
    FIELD_N_XY = FIELD_N_XY
    FIELD_N_Z = FIELD_N_Z

    def get_field_distribution(
        self,
        component: str = "|E|2",
        plane: str = "xz",
        position: float = 0.0,
    ) -> np.ndarray:
        """Extract a 2D solved-field slice from the last simulation.

        Reconstructs fields with torcwa's ``field_xy``/``field_xz``/``field_yz``
        for the last (wavelength, polarization) run. Coordinates follow the
        COMPASS frame (and the Meep adapter): x, y in ``[0, lx) x [0, ly)``,
        z from the bottom of the stack (z=0 at silicon bottom) upward. The
        incident plane wave has unit amplitude (|E0| = 1) and ``Sz`` is the
        z-component of the time-averaged Poynting vector in the same units,
        so the incident flux through a z-plane is ``cos(theta)``: above the
        stack the cell-averaged ``Sz`` equals ``-(1 - R) cos(theta)`` and
        below it ``-T cos(theta)``. ``Sz`` is negative for light travelling
        downward (-z). Fields belong to the last polarization run only.

        If reconstruction fails, a permittivity-based approximation is
        returned instead and a warning is logged.

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
        if self._last_sim is None or self._last_layer_info is None:
            logger.warning("torcwa: no simulation data, returning zeros")
            return np.zeros((FIELD_N_XY, FIELD_N_XY))

        layer_info = self._last_layer_info
        if plane == "xy":
            check_xy_position(layer_info, position)

        try:
            return self._extract_field_from_sim(
                self._last_sim, layer_info, component, plane, position
            )
        except Exception as e:
            logger.warning(
                f"torcwa: field reconstruction failed ({e}); returning a "
                "permittivity-based approximation, not a solved field"
            )
        return self._approximate_field(layer_info, component, plane, position)

    def _extract_field_from_sim(
        self,
        sim,
        layer_info,
        component,
        plane,
        position,
    ) -> np.ndarray:
        """Reconstruct a field slice with torcwa's field_xy/xz/yz.

        torcwa places z=0 at the top (input) interface with z increasing
        downward through the layers in ``layer_info`` order, so a COMPASS
        height ``z`` maps to ``z_top - z``.
        """
        import torch

        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        if self._last_source_amp is None:
            raise RuntimeError("no source amplitude recorded; run a simulation first")

        # Field reconstruction needs the excitation; re-set it in case the QE
        # step was skipped (no photodiodes) or the sim was reused.
        sim.source_planewave(amplitude=self._last_source_amp, direction="forward", notation="ps")

        lx, ly = self._pixel_stack.domain_size
        f64 = torch.float64
        dev = self.device
        xy_axis = torch.arange(FIELD_N_XY, device=dev, dtype=f64) / FIELD_N_XY

        if plane == "xy":
            li, depth = locate_slice(layer_info, position)
            E, H = sim.field_xy(li, xy_axis * lx, xy_axis * ly, z_prop=depth)
        else:
            z_top = stack_z_range(layer_info)[1]
            z_axis = torch.tensor(z_top - column_heights(layer_info), device=dev, dtype=f64)
            if plane == "xz":
                E, H = sim.field_xz(xy_axis * lx, z_axis, float(position))
            else:
                E, H = sim.field_yz(xy_axis * ly, z_axis, float(position))

        E_np = [c.detach().cpu().numpy() for c in E]
        H_np = [c.detach().cpu().numpy() for c in H]
        return component_from_field(E_np, H_np, component)

    def _approximate_field(
        self,
        layer_info,
        component,
        plane,
        position,
    ) -> np.ndarray:
        """Permittivity-based fallback map (see :func:`fields.approximate_field`)."""
        if self._pixel_stack is None:
            raise RuntimeError("pixel_stack is not set; call setup_geometry() first")
        return approximate_field(
            self._pixel_stack, layer_info, self._last_wavelength, component, plane, position
        )

    @staticmethod
    def _component_from_field(E: list, H: list, component: str) -> np.ndarray:
        """Backward-compatible alias of :func:`fields.component_from_field`."""
        return component_from_field(E, H, component)


# Register with factory
SolverFactory.register("torcwa", TorcwaSolver)
