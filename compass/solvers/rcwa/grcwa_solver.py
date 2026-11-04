"""grcwa RCWA solver adapter for COMPASS.

Wraps the grcwa library (GPU-accelerated RCWA with autograd backend).
"""

from __future__ import annotations

import logging

import numpy as np

from compass.core.types import SimulationResult
from compass.geometry.pixel_stack import PixelStack
from compass.solvers.base import SolverBase, SolverFactory
from compass.solvers.rcwa.fields import (
    FIELD_N_XY,
    approximate_field,
    backend_grid_shift,
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

#: Relative |Im(q)|/|q| below which a layer eigenmode is treated as propagating.
_PROPAGATING_REL_TOL = 1e-8


def fix_propagation_branch(q: np.ndarray, rel_tol: float = _PROPAGATING_REL_TOL) -> np.ndarray:
    """Choose the forward branch of layer eigenvalues q = sqrt(eig).

    grcwa picks the branch with ``Im(q) >= 0`` only. For lossless (or nearly
    lossless) layers the eigensolver returns propagating modes with
    round-off imaginary parts of either sign (~1e-17), so roughly half of
    them get flipped to ``Re(q) < 0`` -- a backward wave labelled as
    forward. The S-matrix then mixes forward/backward amplitudes and R/T
    can diverge (observed R ~ 1e60 at oblique incidence on Bayer pixels).

    Rule: modes with ``|Im(q)| <= rel_tol * |q|`` are propagating and get
    ``Re(q) > 0``; all others are evanescent/lossy and get ``Im(q) > 0``.
    Flipping the sign of q keeps the eigenvector valid (same q**2).
    """
    q = np.asarray(q)
    near_real = np.abs(np.imag(q)) <= rel_tol * np.abs(q)
    flip = np.where(near_real, np.real(q) < 0, np.imag(q) < 0)
    return np.where(flip, -q, q)


class _GrcwaFieldSampler:
    """Fast field evaluation on a solved grcwa object.

    grcwa recomputes the full interior S-matrices on every
    ``GetAmplitudes`` call and its ``get_ifft`` allocates one full grid per
    plane wave, which is prohibitive for depth-resolved QE integration.
    This caches the un-translated mode amplitudes per layer and evaluates
    the Fourier series directly. grcwa's z axis points down from the top
    of each layer (like torcwa's), with the incident amplitude |E0| = 1.

    Layer index convention: ``layer_info[i]`` is grcwa layer ``i + 1``
    (layer 0 is the semi-infinite input medium).
    """

    def __init__(self, obj):
        import grcwa.rcwa as rc

        self._rc = rc
        self.obj = obj
        self._amps: dict[int, tuple] = {}
        obj.GetAmplitudes = self._get_amplitudes  # used by Solve_FieldFourier

    def _get_amplitudes(self, which_layer: int, z_offset: float):
        obj, rc = self.obj, self._rc
        if which_layer not in self._amps:
            self._amps[which_layer] = rc.SolveInterior(
                which_layer, obj.a0, obj.bN, obj.q_list, obj.phi_list,
                obj.kp_list, obj.thickness_list,
            )
        ai, bi = self._amps[which_layer]
        return rc.TranslateAmplitudes(
            obj.q_list[which_layer], obj.thickness_list[which_layer], z_offset, ai, bi
        )

    def fourier(self, slice_index: int, depth: float):
        """Fourier coefficients ([ex, ey, ez], [hx, hy, hz]) inside a slice."""
        return self.obj.Solve_FieldFourier(slice_index + 1, depth)

    def e2_on_grid(self, slice_index: int, depth: float, nx: int, ny: int) -> np.ndarray:
        """|E|^2 on the (nx, ny) lattice grid, indexed [x, y]."""
        fe, _ = self.fourier(slice_index, depth)
        G = self.obj.G
        out = np.zeros((nx, ny))
        for coeffs in fe:
            grid = np.zeros((nx, ny), dtype=complex)
            np.add.at(grid, (G[:, 0] % nx, G[:, 1] % ny), np.asarray(coeffs))
            out += np.abs(np.fft.ifft2(grid) * (nx * ny)) ** 2
        return out

    def at_points(self, slice_index: int, depth: float, x: np.ndarray, y: np.ndarray):
        """Fields at points (x, y) (broadcastable arrays, um), Bloch phase included."""
        fe, fh = self.fourier(slice_index, depth)
        kx = np.asarray(self.obj.kx).reshape((-1,) + (1,) * np.ndim(x))
        ky = np.asarray(self.obj.ky).reshape((-1,) + (1,) * np.ndim(x))
        phase = np.exp(1j * (kx * x + ky * y))

        def series(c):
            c = np.asarray(c).reshape((-1,) + (1,) * np.ndim(x))
            return np.sum(c * phase, axis=0)

        return [series(c) for c in fe], [series(c) for c in fh]


class GrcwaSolver(SolverBase):
    """grcwa RCWA solver adapter.

    grcwa uses normalized units and autograd-compatible numpy backend.
    This adapter handles unit conversion and interface mapping.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._last_layer_slices: list | None = None
        self._last_layer_info: list | None = None
        self._last_sampler: _GrcwaFieldSampler | None = None
        self._qe_fallback_used = False
        self._last_wavelength: float | None = None

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")
        self._pixel_stack = pixel_stack
        logger.info(f"grcwa: geometry setup for {pixel_stack.unit_cell} unit cell")

    def setup_source(self, source_config: dict) -> None:
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config
        logger.info(f"grcwa: source setup - {self._source.n_wavelengths} wavelengths")

    def run(self) -> SimulationResult:
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        try:
            import grcwa
        except ImportError as err:
            raise ImportError("grcwa is required. Install with: pip install grcwa") from err

        if self.device != "cpu":
            raise ValueError("grcwa supports CPU execution only")
        params = self.config.get("params", {})
        # grcwa truncates by TOTAL number of plane waves (nG), unlike the
        # per-axis Fourier order [m, m] used by torcwa/meent/fmmax where the
        # total count is (2m+1)^2. Configs should set params.nG explicitly;
        # fourier_order[0] is accepted as a legacy fallback but is NOT
        # equivalent to the same value in the other RCWA solvers.
        fourier_order = params.get("fourier_order")
        if "nG" in params:
            nG = int(params["nG"])
        elif fourier_order is not None:
            nG = int(fourier_order[0])
            logger.warning(
                f"grcwa: params.nG not set; using fourier_order[0]={nG} as the total "
                "plane-wave count. Note grcwa's nG is NOT a per-axis order — set "
                "params.nG explicitly to silence this warning."
            )
        else:
            nG = 9
        n_lens_slices = params.get("n_lens_slices", 30)
        grid_multiplier = params.get("grid_multiplier", 3)

        lx, ly = self._pixel_stack.domain_size
        nx = max(int(params.get("epsilon_grid", 64)), (2 * nG + 1) * grid_multiplier)
        ny = max(int(params.get("epsilon_grid", 64)), (2 * nG + 1) * grid_multiplier)

        pol_runs = self._source.get_polarization_runs()
        all_qe: dict[str, list[float]] = {}
        all_R, all_T, all_A = [], [], []
        self._failed_runs = []
        self._qe_fallback_used = False

        for _wl_idx, wavelength in enumerate(self._source.wavelengths):
            freq = 1.0 / wavelength  # normalized frequency

            layer_slices = self._pixel_stack.get_layer_slices(
                wavelength,
                nx,
                ny,
                n_lens_slices=n_lens_slices,
            )

            R_pol, T_pol, A_pol = [], [], []
            qe_pol_accum: dict[str, list[float]] = {}

            for pol in pol_runs:
                # Set polarization
                if pol == "TE":
                    planewave = {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}
                else:
                    planewave = {"p_amp": 1, "s_amp": 0, "p_phase": 0, "s_phase": 0}

                try:
                    obj = grcwa.obj(
                        nG,
                        L1=[lx, 0],
                        L2=[0, ly],
                        freq=freq,
                        theta=self._source.theta_rad,
                        phi=self._source.phi_rad,
                        verbose=0,
                    )

                    # Input layer (semi-infinite air)
                    obj.Add_LayerUniform(0, 1.0)

                    # Add grid layers (collect eps for later)
                    grid_eps_list = []
                    for s in reversed(layer_slices):
                        obj.Add_LayerGrid(s.thickness, nx, ny)
                        # grcwa reshapes to (Nx, Ny), while LayerSlice is [y, x].
                        grid_eps_list.append(s.eps_grid.T.flatten())

                    # Output layer (semi-infinite air)
                    obj.Add_LayerUniform(0, 1.0)

                    # Init must be called BEFORE GridLayer_geteps (grcwa >= 0.1.2)
                    obj.Init_Setup(Gmethod=0)

                    # Set permittivity for all grid layers (single concatenated call)
                    obj.GridLayer_geteps(np.concatenate(grid_eps_list))
                    # Repair grcwa's Im(q)>=0-only branch choice (see
                    # fix_propagation_branch) before any S-matrix is built.
                    q_list = getattr(obj, "q_list", None)
                    if q_list is not None:
                        obj.q_list = [
                            None if q is None else fix_propagation_branch(q) for q in q_list
                        ]

                    obj.MakeExcitationPlanewave(
                        **planewave,
                        order=0,
                    )
                    R, T = obj.RT_Solve(normalize=1)
                    R = float(np.real(R))
                    T = float(np.real(T))
                    A = 1.0 - R - T
                    sampler = _GrcwaFieldSampler(obj)

                    if R > 1.0 or R < 0.0:
                        logger.warning(
                            f"grcwa: unphysical R={R:.4f} at λ={wavelength:.4f}um, "
                            f"pol={pol}, nG={nG}. Check convergence in nG and "
                            "epsilon_grid; eigenmode branches are already repaired."
                        )

                except Exception as e:
                    logger.error(f"grcwa failed at λ={wavelength:.4f}um: {e}")
                    self._record_failed_run(wavelength, pol, e)
                    R = T = A = float("nan")
                    sampler = None

                R_pol.append(R)
                T_pol.append(T)
                A_pol.append(A)

                self._last_layer_slices = layer_slices
                self._last_layer_info = list(reversed(layer_slices))
                self._last_wavelength = wavelength
                self._last_sampler = sampler

                if np.isnan(A) or sampler is None:
                    pixel_qe = self._nan_pixel_qe()
                elif not self._pixel_stack.photodiodes:
                    pixel_qe = {}
                else:
                    pixel_qe = self._compute_qe_with_fallback(sampler, layer_slices, wavelength, A)
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

                warnings.warn(f"grcwa: NaN/Inf detected in {arr_name} output", stacklevel=2)

        return SimulationResult(
            qe_per_pixel={k: np.array(v) for k, v in all_qe.items()},
            wavelengths=self._source.wavelengths,
            reflection=result_arrays["reflection"],
            transmission=result_arrays["transmission"],
            absorption=result_arrays["absorption"],
            metadata={
                "solver_name": "grcwa",
                "nG": nG,
                # Field integration vs. the eps''-weight approximation is a
                # QE methodology difference, not solver accuracy.
                "qe_method": (
                    "eps_imag_weight_fallback" if self._qe_fallback_used else "field_integration"
                ),
                "device": self.device,
                **self._failure_metadata(),
            },
        )

    def _compute_qe_with_fallback(
        self,
        sampler: _GrcwaFieldSampler,
        layer_slices,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
        """Field-integrated QE, falling back to eps''-weighting on failure."""
        try:
            return self._compute_per_pixel_qe_fields(sampler, wavelength, total_absorption)
        except Exception as e:
            logger.warning(
                f"grcwa: field-based QE failed at λ={wavelength:.4f}um ({e}); "
                "falling back to eps''-weight approximation"
            )
            self._qe_fallback_used = True
            return self._compute_per_pixel_qe(layer_slices, wavelength, total_absorption)

    def _compute_per_pixel_qe_fields(
        self,
        sampler: _GrcwaFieldSampler,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
        """Per-pixel QE by integrating eps'' |E|^2 over photodiode volumes.

        Uses the QE definition shared with torcwa
        (:func:`compass.solvers.rcwa.fields.integrate_photodiode_qe`).
        """
        if self._pixel_stack is None or self._source is None:
            raise RuntimeError("Call setup_geometry() and setup_source() before run()")
        if self._last_layer_info is None:
            raise RuntimeError("no layer data recorded")

        def e2_dz(li: int, s) -> np.ndarray:
            n0, n1 = s.eps_grid.shape  # [y, x]; grcwa grids are [x, y]
            depths, dz = slice_depth_samples(s.thickness)
            acc = np.zeros((n1, n0))
            for depth in depths:
                acc += sampler.e2_on_grid(li, float(depth), n1, n0)
            return acc.T * dz

        return integrate_photodiode_qe(
            self._pixel_stack,
            self._last_layer_info,
            wavelength,
            self._source.theta_rad,
            e2_dz,
            total_absorption,
            "grcwa",
        )

    def _compute_per_pixel_qe(
        self,
        layer_slices,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
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

        Same conventions as the torcwa adapter: x, y in ``[0, lx) x [0, ly)``,
        z up from the stack bottom, |E0| = 1, and ``Sz`` the Poynting
        z-component (negative for downward flow; incident flux
        ``cos(theta)``). Fields belong to the last polarization run.

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
        if self._last_sampler is None or self._last_layer_info is None:
            logger.warning("grcwa: no simulation data, returning zeros")
            return np.zeros((FIELD_N_XY, FIELD_N_XY))
        if plane == "xy":
            check_xy_position(self._last_layer_info, position)
        try:
            return self._solved_field(component, plane, position)
        except Exception as e:
            logger.warning(
                f"grcwa: field reconstruction failed ({e}); returning a "
                "permittivity-based approximation, not a solved field"
            )
        return self._approximate_field(component, plane, position)

    def _solved_field(self, component: str, plane: str, position: float) -> np.ndarray:
        if self._pixel_stack is None or self._last_sampler is None:
            raise RuntimeError("no simulation data")
        if self._last_layer_info is None:
            raise RuntimeError("no layer data recorded")
        sampler = self._last_sampler
        layer_info = self._last_layer_info
        lx, ly = self._pixel_stack.domain_size
        dx, dy = backend_grid_shift(self._pixel_stack, layer_info)
        u = np.arange(FIELD_N_XY) / FIELD_N_XY

        if plane == "xy":
            li, depth = locate_slice(layer_info, position)
            X, Y = np.meshgrid(u * lx - dx, u * ly - dy, indexing="ij")
            E, H = sampler.at_points(li, depth, X, Y)
            return component_from_field(E, H, component)

        heights = column_heights(layer_info)
        line = u * (lx if plane == "xz" else ly)
        x = line - dx if plane == "xz" else np.full_like(line, position - dx)
        y = np.full_like(line, position - dy) if plane == "xz" else line - dy
        out = np.zeros((FIELD_N_XY, len(heights)))
        for zi, z in enumerate(heights):
            li, depth = locate_slice(layer_info, float(z))
            E, H = sampler.at_points(li, depth, x, y)
            out[:, zi] = component_from_field(E, H, component)
        return out

    def _approximate_field(
        self,
        component: str,
        plane: str,
        position: float,
    ) -> np.ndarray:
        """Permittivity-based fallback map (see :func:`fields.approximate_field`)."""
        if self._pixel_stack is None or self._last_layer_info is None:
            return np.zeros((FIELD_N_XY, FIELD_N_XY))
        return approximate_field(
            self._pixel_stack, self._last_layer_info, self._last_wavelength,
            component, plane, position,
        )


SolverFactory.register("grcwa", GrcwaSolver)
