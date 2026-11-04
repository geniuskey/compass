"""meent RCWA solver adapter for COMPASS.

Wraps the meent library (multi-backend RCWA: NumPy, JAX, PyTorch).
"""

from __future__ import annotations

import logging
from typing import Any

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


def _dfs_work_grid_shape(shape_yx, fourier_order, enhanced_dfs):
    """Installed meent raster-DFS repeat rule, before the three Fourier transforms."""
    result = []
    for size, order in zip(shape_yx, reversed(fourier_order), strict=True):
        minimum = (4 * order + 1) * size if enhanced_dfs else 4 * order + 1
        result.append(size * (minimum // size + 1) if size < minimum else size)
    return result


def _to_numpy(a) -> np.ndarray:
    """Convert a NumPy/JAX/torch array to NumPy."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a)


class _MeentFieldSampler:
    """Field evaluation on a solved 2D meent (enhanced TMM) object.

    Mirrors meent's ``field_dist_2d`` (matching it to round-off) but caches
    the per-layer mode coefficients and evaluates at arbitrary depths and
    points, which the depth-resolved QE integral needs. Arrays are converted
    to NumPy, so every meent backend is supported.

    Conventions, validated against the flux and Poynting-theorem invariants:

    * meent uses ``exp(+j omega t)`` and is fed ``n - ik``; its fields are
      complex conjugates of COMPASS fields (``|E|^2`` and ``Re(E x H*)`` are
      unchanged). Its H carries the opposite sign of the torcwa/grcwa
      backend frame, so H is negated here: ``Re(Ex Hy* - Ey Hx*)`` is then
      the downward power flux, and the incident flux equals ``cos(theta)``.
    * Lengths inside meent are nm; the public methods take um.
    * Layer ``i`` (top first) is meent layer ``i``; depth is measured down
      from that layer's top.
    * meent's eps sample ``(row a, col b)`` sits at ``((b + o) lx/n1,
      (a + o) ly/n0)``, where the offset ``o`` follows from the raster
      Fourier transform (cell centre for CFS and repeated-pixel DFS).
      COMPASS samples cell centres, so points are shifted by ``(0.5 - o)``
      cells before evaluation.
    * meent factorizes the permittivity with Li's inverse rule, under which
      the truncated E series rings strongly at metal / high-contrast
      interfaces (the real-space ``eps''|E|^2`` integral overshoots the
      absorbed power by 2x in a metal grid). The real-space E is therefore
      reconstructed from the displacement, ``E = D / eps(r)``, with
      ``D = [[eps]] E`` taken from meent's own convolution matrices.
    """

    def __init__(self, mee, fourier_type: int = 0, enhanced_dfs: bool = True):
        wl_nm = float(np.real(_to_numpy(mee.wavelength)))
        self.k0 = 2.0 * np.pi / wl_nm  # nm^-1
        kx, ky = mee.get_kx_ky_vector(wavelength=mee.wavelength)
        self.kx = _to_numpy(kx).astype(complex)
        self.ky = _to_numpy(ky).astype(complex)
        self.ffx, self.ffy = len(self.kx), len(self.ky)
        self.kx_flat = np.tile(self.kx, self.ffy)  # index = iy * ffx + ix
        self.ky_flat = np.repeat(self.ky, self.ffx)
        self.fto = (int(mee.fto[0]), int(mee.fto[1]))
        self.fourier_type = fourier_type
        self.enhanced_dfs = enhanced_dfs
        self.use_pinv = bool(getattr(mee, "use_pinv", False))
        self.period_nm = (float(mee.period[0]), float(mee.period[1]))
        # meent permittivity (n - ik)^2 per layer, [layer, y, x]
        self.eps = _to_numpy(mee.ucell).astype(complex) ** 2
        n0, n1 = self.eps.shape[1:]
        if fourier_type == 0:
            work = _dfs_work_grid_shape([n0, n1], list(self.fto), enhanced_dfs)
            offset_y, offset_x = ((1.0 - n / w) / 2.0 for n, w in zip((n0, n1), work, strict=True))
        else:
            offset_y = offset_x = 0.5
        # COMPASS -> meent shift (nm) of the sample frame
        self._shift = (
            (0.5 - offset_x) * self.period_nm[0] / n1,
            (0.5 - offset_y) * self.period_nm[1] / n0,
        )
        self._conv: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        T = _to_numpy(mee.T1)[0]  # (2 ffxy, 1): incidence set by mee.pol/psi
        self._layers = []
        for info in mee.layer_info_list[::-1]:  # top layer first
            epz_conv_i, W, V, q, d_arr, A_i, B = (_to_numpy(x) for x in info)
            d = float(np.real(d_arr))
            X = np.exp(-self.k0 * q * d)[:, None]
            c_plus = T[:, 0]
            c_minus = (B @ (A_i @ (X * T)))[:, 0]
            self._layers.append((epz_conv_i, W, V, q, d, c_plus, c_minus))
            T = A_i @ (X * T)

    def _transverse_conv(self, li: int) -> tuple[np.ndarray, np.ndarray]:
        """meent's (epx, epy) convolution matrices for layer ``li``."""
        if li not in self._conv:
            from meent.on_numpy.emsolver import convolution_matrix as cm

            cell = np.sqrt(self.eps[li : li + 1])
            if self.fourier_type == 0:
                epx, epy, _ = cm.to_conv_mat_raster_discrete(
                    cell, *self.fto, type_complex=np.complex128,
                    enhanced_dfs=self.enhanced_dfs, use_pinv=self.use_pinv,
                )
            else:
                epx, epy, _ = cm.to_conv_mat_raster_continuous(
                    cell, *self.fto, type_complex=np.complex128, use_pinv=self.use_pinv,
                )
            self._conv[li] = (np.asarray(epx[0]), np.asarray(epy[0]))
        return self._conv[li]

    def fourier(self, slice_index: int, depth_um):
        """Fourier coefficients at the given depths, each (n_depth, ffxy).

        Returns:
            ``([ex, ey, ez], [hx, hy, hz], [dx, dy, dz])``; ``d`` is the
            displacement (``eps`` times E in meent's units).
        """
        epz_conv_i, W, V, q, d, c_plus, c_minus = self._layers[slice_index]
        z = np.atleast_1d(np.asarray(depth_um, dtype=float))[:, None] * 1000.0
        a_plus = np.exp(-self.k0 * q[None, :] * z) * c_plus[None, :]
        a_minus = np.exp(self.k0 * q[None, :] * (z - d)) * c_minus[None, :]
        n = self.ffx * self.ffy
        S = (a_plus + a_minus) @ W.T
        U = (a_minus - a_plus) @ V.T
        sx, sy = S[:, :n], S[:, n:]
        ux, uy = U[:, :n], U[:, n:]
        dz = -1j * (self.kx_flat * uy - self.ky_flat * ux)
        sz = dz @ epz_conv_i.T
        uz = -1j * (self.kx_flat * sy - self.ky_flat * sx)
        epx, epy = self._transverse_conv(slice_index)
        return [sx, sy, sz], [-1j * ux, -1j * uy, -1j * uz], [sx @ epx.T, sy @ epy.T, dz]

    def power_down(self, slice_index: int, depth_um: float) -> float:
        """Cell-averaged downward Poynting flux (incident flux = cos(theta))."""
        (ex, ey, _), (hx, hy, _), _ = self.fourier(slice_index, np.array([depth_um]))
        return float(np.sum(np.real(ex * np.conj(hy) - ey * np.conj(hx))))

    def _native(self, x_um, y_um):
        x = np.asarray(x_um, dtype=float) * 1000.0 - self._shift[0]
        y = np.asarray(y_um, dtype=float) * 1000.0 - self._shift[1]
        return x, y

    def _eps_at(self, slice_index: int, x_um, y_um) -> np.ndarray:
        """meent permittivity of the grid cell containing COMPASS (x, y)."""
        eps = self.eps[slice_index]
        n0, n1 = eps.shape
        lx, ly = self.period_nm[0] / 1000.0, self.period_nm[1] / 1000.0
        ix = np.floor(np.asarray(x_um) / lx * n1).astype(int) % n1
        iy = np.floor(np.asarray(y_um) / ly * n0).astype(int) % n0
        return np.asarray(eps[iy, ix])

    def e2_on_grid(self, slice_index: int, depths_um, x_um, y_um) -> np.ndarray:
        """Sum over depths of |E|^2 = |D/eps|^2 on the grid, indexed [y, x]."""
        _, _, fd = self.fourier(slice_index, depths_um)
        xn, yn = self._native(x_um, y_um)
        px = np.exp(-1j * self.k0 * np.outer(xn, self.kx))  # (nx, ffx)
        py = np.exp(-1j * self.k0 * np.outer(yn, self.ky))  # (ny, ffy)
        Y, X = np.meshgrid(np.asarray(y_um), np.asarray(x_um), indexing="ij")
        inv_eps2 = 1.0 / np.abs(self._eps_at(slice_index, X, Y)) ** 2
        out = np.zeros((len(yn), len(xn)))
        for c in fd:
            C = c.reshape(-1, self.ffy, self.ffx)
            out += np.sum(np.abs(np.einsum("zij,ai,bj->zab", C, py, px)) ** 2, axis=0)
        return np.asarray(out * inv_eps2)

    def at_points(
        self, slice_index: int, depth_um: float, x_um, y_um, displacement: bool = True
    ):
        """Fields ([Ex, Ey, Ez], [Hx, Hy, Hz]) at COMPASS points (x, y) in um.

        With ``displacement`` (default) E is ``D / eps(r)``; otherwise the
        raw truncated E series, whose plane average of ``Re(E x H*)`` equals
        the Fourier-space flux exactly (used for ``Sz``).
        """
        fe, fh, fd = self.fourier(slice_index, np.array([depth_um]))
        x, y = self._native(x_um, y_um)
        shape = (-1,) + (1,) * x.ndim
        phase = np.exp(
            -1j * self.k0 * (self.kx_flat.reshape(shape) * x + self.ky_flat.reshape(shape) * y)
        )

        def series(c):
            return np.sum(c[0].reshape(shape) * phase, axis=0)

        H = [series(c) for c in fh]
        if not displacement:
            return [series(c) for c in fe], H
        eps = self._eps_at(slice_index, x_um, y_um)
        return [series(c) / eps for c in fd], H


class MeentSolver(SolverBase):
    """meent RCWA solver adapter.

    Supports NumPy, JAX, and PyTorch backends via config.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__(config, device)
        self._source: PlanewaveSource | None = None
        self._backend = config.get("params", {}).get("backend", "numpy")
        self._last_layer_slices: list | None = None
        self._last_layer_info: list | None = None
        self._last_sampler: _MeentFieldSampler | None = None
        self._last_wavelength: float | None = None
        self._qe_fallback_used = False

    def setup_geometry(self, pixel_stack: PixelStack) -> None:
        if pixel_stack is None:
            raise ValueError("pixel_stack must not be None")
        if not pixel_stack.layers:
            raise ValueError("pixel_stack must have at least one layer")
        self._pixel_stack = pixel_stack
        logger.info(f"meent: geometry setup, backend={self._backend}")

    def setup_source(self, source_config: dict) -> None:
        self._source = PlanewaveSource.from_config(source_config)
        if self._source.n_wavelengths == 0:
            raise ValueError("wavelengths array must not be empty")
        if np.any(self._source.wavelengths <= 0):
            raise ValueError("all wavelengths must be positive")
        self._source_config = source_config
        logger.info(f"meent: source setup - {self._source.n_wavelengths} wavelengths")

    def run(self) -> SimulationResult:
        if self._pixel_stack is None:
            raise RuntimeError("Call setup_geometry() before run()")
        if self._source is None:
            raise RuntimeError("Call setup_source() before run()")

        try:
            import meent
        except ImportError as err:
            raise ImportError("meent is required. Install with: pip install meent") from err

        if self._backend == "numpy" and self.device != "cpu":
            raise ValueError("meent/numpy supports CPU execution only")
        if self._backend == "jax":
            import jax

            target = "gpu" if self.device.startswith("cuda") else "cpu"
            jax.config.update(
                "jax_default_device", jax.devices(target)[cuda_device_index(self.device)]
            )

        params = self.config.get("params", {})
        fourier_order = params.get("fourier_order", [9, 9])
        fourier_type = params.get("fourier_type", 0)
        if fourier_type not in (0, 1):
            raise ValueError("meent fourier_type must be 0 (DFS) or 1 (CFS)")
        enhanced_dfs = params.get("enhanced_dfs", True)
        if not isinstance(enhanced_dfs, bool):
            raise ValueError("meent enhanced_dfs must be a boolean")

        lx, ly = self._pixel_stack.domain_size
        # Zero disables automatic oversampling for independent grid/order studies.
        grid_multiplier = params.get("grid_multiplier", 3)
        nx = max(int(params.get("epsilon_grid", 64)), (2 * fourier_order[0] + 1) * grid_multiplier)
        ny = max(int(params.get("epsilon_grid", 64)), (2 * fourier_order[1] + 1) * grid_multiplier)

        pol_runs = self._source.get_polarization_runs()
        self._failed_runs = []
        self._qe_fallback_used = False
        all_qe: dict[str, list[float]] = {}
        all_R, all_T, all_A = [], [], []

        for _wl_idx, wavelength in enumerate(self._source.wavelengths):
            layer_slices = self._pixel_stack.get_layer_slices(
                wavelength, nx, ny, n_lens_slices=params.get("n_lens_slices", 30),
            )

            R_pol, T_pol, A_pol = [], [], []
            qe_pol_accum: dict[str, list[float]] = {}

            for pol in pol_runs:
                try:
                    # meent backend selection
                    backend_id = {"numpy": 0, "jax": 1, "torch": 2}.get(self._backend, 0)
                    device_kwargs: dict[str, Any] = {}
                    if backend_id == 1:
                        device_kwargs["device"] = 1 if self.device.startswith("cuda") else 0
                    if backend_id == 2:
                        import torch

                        device_kwargs["device"] = torch.device(self.device)

                    mee = meent.call_mee(
                        backend=backend_id,
                        **device_kwargs,
                        pol=0 if pol == "TE" else 1,
                        n_top=1.0,
                        n_bot=1.0,
                        theta=self._source.theta_rad,
                        phi=self._source.phi_rad,
                        fto=fourier_order,
                        fourier_type=fourier_type,
                        enhanced_dfs=enhanced_dfs,
                        period=[lx * 1000, ly * 1000],  # meent uses nm
                        wavelength=wavelength * 1000,  # meent uses nm
                        thickness=[s.thickness * 1000 for s in reversed(layer_slices)],
                        type_complex=np.complex128,
                    )

                    # Set permittivity for each layer
                    ucell_list = []
                    for s in reversed(layer_slices):
                        ucell_list.append(s.eps_grid[np.newaxis, :, :])

                    # meent uses n-ik for passive media; COMPASS uses n+ik.
                    # Convert both the optical quantity and time convention.
                    mee.ucell = np.conj(np.sqrt(np.concatenate(ucell_list, axis=0)))
                    result = mee.conv_solve()
                    de_ri = result.de_ri
                    de_ti = result.de_ti
                    if backend_id == 2:
                        # NumPy reductions cannot operate directly on torch tensors,
                        # including tensors resident on a CUDA device.
                        de_ri = de_ri.detach().cpu().numpy()
                        de_ti = de_ti.detach().cpu().numpy()

                    R = float(np.sum(de_ri))
                    T = float(np.sum(de_ti))
                    if R + T > 1.0 + 0.01:
                        logger.warning(
                            f"meent: R+T={R + T:.4f} > 1 at λ={wavelength:.4f}um "
                            f"(numerical instability for multi-layer 2D stack)"
                        )
                    A = 1.0 - R - T
                    try:
                        sampler: _MeentFieldSampler | None = _MeentFieldSampler(
                            mee, fourier_type, enhanced_dfs
                        )
                    except Exception as e:  # field data unavailable -> QE fallback
                        logger.warning(f"meent: field reconstruction setup failed ({e})")
                        sampler = None

                except Exception as e:
                    logger.error(f"meent failed at λ={wavelength:.4f}um: {e}")
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

                if np.isnan(A):
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

                warnings.warn(f"meent: NaN/Inf detected in {arr_name} output", stacklevel=2)

        return SimulationResult(
            qe_per_pixel={k: np.array(v) for k, v in all_qe.items()},
            wavelengths=self._source.wavelengths,
            reflection=result_arrays["reflection"],
            transmission=result_arrays["transmission"],
            absorption=result_arrays["absorption"],
            metadata={
                "solver_name": "meent",
                "backend": self._backend,
                "fourier_order": fourier_order,
                "epsilon_grid_requested": int(params.get("epsilon_grid", 64)),
                "epsilon_grid_shape_yx": [ny, nx],
                "grid_multiplier": grid_multiplier,
                "fourier_type": fourier_type,
                "fourier_algorithm": "raster_dfs" if fourier_type == 0 else "raster_cfs",
                "enhanced_dfs": enhanced_dfs,
                "fourier_work_grid_shape_yx": (
                    _dfs_work_grid_shape([ny, nx], fourier_order, enhanced_dfs)
                    if fourier_type == 0 else None
                ),
                "fourier_work_grid_shape_source": (
                    "derived from meent raster DFS repeat rule; CFS has no FFT work grid"
                ),
                # Field integration vs. the eps''-weight approximation is a
                # QE methodology difference, not solver accuracy.
                "qe_method": (
                    "eps_imag_weight_fallback" if self._qe_fallback_used else "field_integration"
                ),
                **self._failure_metadata(),
            },
        )

    def _compute_qe_with_fallback(
        self,
        sampler: _MeentFieldSampler | None,
        layer_slices,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
        """Field-integrated QE, falling back to eps''-weighting on failure."""
        try:
            if sampler is None:
                raise RuntimeError("no field data")
            return self._compute_per_pixel_qe_fields(sampler, wavelength, total_absorption)
        except Exception as e:
            logger.warning(
                f"meent: field-based QE failed at λ={wavelength:.4f}um ({e}); "
                "falling back to eps''-weight approximation"
            )
            self._qe_fallback_used = True
            return self._compute_per_pixel_qe(layer_slices, wavelength, total_absorption)

    def _compute_per_pixel_qe_fields(
        self,
        sampler: _MeentFieldSampler,
        wavelength: float,
        total_absorption: float,
    ) -> dict:
        """Per-pixel QE by integrating eps'' |E|^2 over photodiode volumes.

        Uses the QE definition shared with the other RCWA adapters
        (:func:`compass.solvers.rcwa.fields.integrate_photodiode_qe`).
        """
        if self._pixel_stack is None or self._source is None:
            raise RuntimeError("Call setup_geometry() and setup_source() before run()")
        if self._last_layer_info is None:
            raise RuntimeError("no layer data recorded")
        lx, ly = self._pixel_stack.domain_size

        def e2_dz(li: int, s) -> np.ndarray:
            n0, n1 = s.eps_grid.shape  # [y, x]
            depths, dz = slice_depth_samples(s.thickness)
            x = (np.arange(n1) + 0.5) * (lx / n1)  # COMPASS cell centres
            y = (np.arange(n0) + 0.5) * (ly / n0)
            return sampler.e2_on_grid(li, depths, x, y) * dz

        cos_theta = float(np.cos(self._source.theta_rad))

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
            "meent",
            slice_absorption=absorbed,
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

        Same conventions as the torcwa/grcwa adapters: x, y in
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
            logger.warning("meent: no simulation data, returning zeros")
            return np.zeros((FIELD_N_XY, FIELD_N_XY))
        if plane == "xy":
            check_xy_position(self._last_layer_info, position)
        try:
            if self._last_sampler is None:
                raise RuntimeError("no field data from the last run")
            return self._solved_field(self._last_sampler, component, plane, position)
        except Exception as e:
            logger.warning(
                f"meent: field reconstruction failed ({e}); returning a "
                "permittivity-based approximation, not a solved field"
            )
        return approximate_field(
            self._pixel_stack, self._last_layer_info, self._last_wavelength,
            component, plane, position,
        )

    def _solved_field(
        self, sampler: _MeentFieldSampler, component: str, plane: str, position: float
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


SolverFactory.register("meent", MeentSolver)
